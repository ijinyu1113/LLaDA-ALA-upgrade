"""
Scaled Benchmark Evaluation for ALA-LLaDA
==========================================
Runs GSM8K, MATH, ARC-Challenge, GPQA Diamond, BBH (hardest subtasks)
with checkpointing for crash recovery.

Usage:
    python run_benchmarks.py --benchmarks gsm8k math arc gpqa bbh
    python run_benchmarks.py --benchmarks arc gpqa bbh   # fast ones first
    python run_benchmarks.py --benchmarks gsm8k --resume # overnight run
"""

import argparse
import json
import os
import re
import sys
import tempfile
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, concatenate_datasets

from models import ALALLaDA, MASK_TOKEN_ID, ALPHA_BASE, ALPHA_SCALE
from generation_utils import generate

INFERENCE_ALPHA = 0.02


# ============================================================
# Shared Utilities
# ============================================================

def checkpoint_save(path, data):
    """Atomic JSON write — write to temp file, then rename."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)
    except Exception:
        os.unlink(tmp)
        raise


def checkpoint_load(path):
    """Load checkpoint or return empty dict."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def format_time(seconds):
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    return f"{m}m {s:02d}s"


# --- Answer extraction ---

def extract_answer_gsm8k(text):
    if "####" in text:
        after = text.split("####")[-1].strip()
        match = re.match(r'-?\d[\d,]*\.?\d*', after)
        if match:
            return match.group().rstrip('.').replace(',', '')
    numbers = re.findall(r'-?\d[\d,]*\.?\d*', text)
    if numbers:
        return numbers[-1].rstrip('.').replace(',', '')
    return ""


def extract_gold_gsm8k(answer_text):
    if "####" in answer_text:
        after = answer_text.split("####")[-1].strip()
        return ''.join(c for c in after if c.isdigit() or c == '-')
    return ""


def extract_boxed(text):
    """Extract answer from \\boxed{...} format used in MATH."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        idx = text.rfind("boxed{")
        if idx == -1:
            return text.strip().split()[-1] if text.strip() else ""
        idx += 6
    else:
        idx += 7
    depth = 1
    end = idx
    while end < len(text) and depth > 0:
        if text[end] == '{':
            depth += 1
        elif text[end] == '}':
            depth -= 1
        end += 1
    return text[idx:end-1].strip() if depth == 0 else ""


def extract_answer_arc(text):
    """Extract first letter A-E from response."""
    match = re.search(r'([A-E])', text.upper())
    return match.group(1) if match else ""


# ============================================================
# Model Loading
# ============================================================

def load_model():
    model_id = 'GSAI-ML/LLaDA-8B-Instruct'
    print("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = ALALLaDA(base_model).to(torch.bfloat16)
    device = next(base_model.parameters()).device
    model.router.to(device)

    weights_path = "amip_router_best.pt"
    if os.path.exists(weights_path):
        model.router.load_state_dict(
            torch.load(weights_path, map_location=device),
            strict=False
        )
        print(f"Router loaded from {weights_path}")
    else:
        print("WARNING: Using random router weights.")

    model.eval()
    print(f"Inference alpha: {INFERENCE_ALPHA}")
    return model, tokenizer


# ============================================================
# Generic eval loop helper
# ============================================================

def _run_eval_loop(items, model, tokenizer, args, benchmark_name,
                   extract_fn, match_fn, gen_length=256, steps=256,
                   block_length=32, progress_every=50):
    """Generic eval loop with checkpointing.

    items: list of (idx, prompt_ids, gold, meta_dict)
    extract_fn: response_text -> predicted answer
    match_fn: (pred, gold) -> bool
    """
    num_samples = len(items)
    ckpt_path = os.path.join(args.checkpoint_dir, f"{benchmark_name}_checkpoint.json")
    ckpt = checkpoint_load(ckpt_path) if args.resume else {}
    completed = set(ckpt.get("completed_indices", []))
    per_sample = ckpt.get("per_sample", {})

    if completed:
        print(f"  Resuming: {len(completed)} samples already done")

    t0 = time.time()
    done_this_run = 0

    for item_idx, (idx, ids, gold, meta) in enumerate(items):
        if idx in completed:
            continue

        result = {"gold": gold}
        result.update(meta)

        for mode in ["Baseline", "Router"]:
            use_router = (mode == "Router")
            alpha = INFERENCE_ALPHA if use_router else None
            torch.manual_seed(42)
            out = generate(model, ids, steps=steps, gen_length=gen_length,
                           block_length=block_length,
                           use_router=use_router, temp=0.0, alpha=alpha)
            response = tokenizer.decode(
                out[0, ids.shape[1]:], skip_special_tokens=True
            ).strip()
            pred = extract_fn(response)
            is_correct = match_fn(pred, gold)
            result[f"pred_{mode.lower()}"] = pred
            result[f"correct_{mode.lower()}"] = is_correct

        per_sample[str(idx)] = result
        completed.add(idx)
        done_this_run += 1

        n_done = len(completed)
        if n_done % progress_every == 0 or done_this_run <= 2:
            elapsed = time.time() - t0
            rate = done_this_run / elapsed if elapsed > 0 else 0
            remaining = (num_samples - n_done) / rate if rate > 0 else 0
            base_correct = sum(1 for v in per_sample.values() if v.get("correct_baseline"))
            router_correct = sum(1 for v in per_sample.values() if v.get("correct_router"))
            n_scored = sum(1 for v in per_sample.values() if "correct_baseline" in v)
            if n_scored > 0:
                print(f"  [{n_done}/{num_samples}] "
                      f"Base: {base_correct}/{n_scored} ({base_correct/n_scored:.1%})  "
                      f"Router: {router_correct}/{n_scored} ({router_correct/n_scored:.1%})  "
                      f"ETA: {format_time(remaining)}", flush=True)

        if done_this_run % args.save_every == 0:
            checkpoint_save(ckpt_path, {
                "completed_indices": sorted(completed),
                "per_sample": per_sample,
            })

    checkpoint_save(ckpt_path, {
        "completed_indices": sorted(completed),
        "per_sample": per_sample,
    })

    n_scored = sum(1 for v in per_sample.values() if "correct_baseline" in v)
    base_correct = sum(1 for v in per_sample.values() if v.get("correct_baseline"))
    router_correct = sum(1 for v in per_sample.values() if v.get("correct_router"))
    base_acc = base_correct / n_scored if n_scored > 0 else 0
    router_acc = router_correct / n_scored if n_scored > 0 else 0

    print(f"\n  {benchmark_name} Final Results (n={n_scored}):")
    print(f"  Baseline: {base_acc:.4f} ({base_correct}/{n_scored})")
    print(f"  Router:   {router_acc:.4f} ({router_correct}/{n_scored})")
    print(f"  Delta:    {router_acc - base_acc:+.4f}")

    return {
        "Baseline": {"accuracy": base_acc, "correct": base_correct, "total": n_scored},
        "Router": {"accuracy": router_acc, "correct": router_correct, "total": n_scored},
        "per_sample": per_sample,
    }


# ============================================================
# GSM8K (n=200)
# ============================================================

@torch.no_grad()
def eval_gsm8k(model, tokenizer, args, n=200):
    print(f"\n{'='*60}", flush=True)
    print(f"GSM8K (n={n}, α={INFERENCE_ALPHA})", flush=True)
    print(f"{'='*60}", flush=True)

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    items = []
    for idx in range(min(n, len(dataset))):
        gold = extract_gold_gsm8k(dataset[idx]["answer"])
        if not gold:
            continue
        prompt = (f"Solve this math problem step by step. "
                  f"Give your final answer after ####.\n\n"
                  f"Question: {dataset[idx]['question']}\n\nSolution:")
        ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=512)["input_ids"].to(model.device)
        items.append((idx, ids, gold, {}))

    print(f"  Valid samples: {len(items)}", flush=True)
    return _run_eval_loop(items, model, tokenizer, args, "gsm8k",
                          extract_fn=extract_answer_gsm8k,
                          match_fn=lambda p, g: p == g,
                          gen_length=256, steps=256, progress_every=50)


# ============================================================
# MATH (n=200)
# ============================================================

@torch.no_grad()
def eval_math(model, tokenizer, args, n=200):
    print(f"\n{'='*60}", flush=True)
    print(f"MATH (n={n}, α={INFERENCE_ALPHA})", flush=True)
    print(f"{'='*60}", flush=True)

    subjects = ["algebra", "counting_and_probability", "geometry",
                "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
    parts = [load_dataset("EleutherAI/hendrycks_math", subj, split="test")
             for subj in subjects]
    dataset = concatenate_datasets(parts)

    items = []
    for idx in range(min(n, len(dataset))):
        sample = dataset[idx]
        gold = extract_boxed(sample["solution"])
        if not gold:
            continue
        level = sample.get("level", "Unknown")
        prompt = (f"Solve this math problem. Put your final answer in \\boxed{{}}.\n\n"
                  f"Problem: {sample['problem']}\n\nSolution:")
        ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=512)["input_ids"].to(model.device)
        items.append((idx, ids, gold, {"level": level}))

    print(f"  Valid samples: {len(items)}", flush=True)
    return _run_eval_loop(items, model, tokenizer, args, "math",
                          extract_fn=extract_boxed,
                          match_fn=lambda p, g: p.strip().rstrip('.') == g.strip().rstrip('.'),
                          gen_length=256, steps=256, progress_every=50)


# ============================================================
# ARC-Challenge (n=200)
# ============================================================

@torch.no_grad()
def eval_arc(model, tokenizer, args, n=200):
    print(f"\n{'='*60}", flush=True)
    print(f"ARC-Challenge (n={n}, α={INFERENCE_ALPHA})", flush=True)
    print(f"{'='*60}", flush=True)

    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    items = []
    for idx in range(min(n, len(dataset))):
        sample = dataset[idx]
        choices = sample["choices"]
        choice_lines = [f"{label}) {text}"
                        for label, text in zip(choices["label"], choices["text"])]
        prompt = (f"Question: {sample['question']}\n"
                  f"{chr(10).join(choice_lines)}\n\n"
                  f"Answer:")
        ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=512)["input_ids"].to(model.device)
        items.append((idx, ids, sample["answerKey"].upper(), {}))

    print(f"  Valid samples: {len(items)}", flush=True)
    return _run_eval_loop(items, model, tokenizer, args, "arc",
                          extract_fn=extract_answer_arc,
                          match_fn=lambda p, g: p == g,
                          gen_length=32, steps=32, progress_every=50)


# ============================================================
# GPQA Diamond (n=198, all of them)
# ============================================================

@torch.no_grad()
def eval_gpqa(model, tokenizer, args):
    print(f"\n{'='*60}", flush=True)
    print(f"GPQA Diamond (α={INFERENCE_ALPHA})", flush=True)
    print(f"{'='*60}", flush=True)

    dataset = load_dataset("hendrydong/gpqa_diamond_mc", split="test")
    items = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        prompt = (f"Answer this graduate-level question. "
                  f"Put your final answer in \\boxed{{}}.\n\n"
                  f"Question: {sample['Question']}\n"
                  f"(A) {sample['choice_A']}\n"
                  f"(B) {sample['choice_B']}\n"
                  f"(C) {sample['choice_C']}\n"
                  f"(D) {sample['choice_D']}\n\n"
                  f"Answer:")
        ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=512)["input_ids"].to(model.device)
        gold = sample["answer"]  # "A", "B", "C", or "D"
        items.append((idx, ids, gold.upper(), {}))

    def extract_gpqa(text):
        boxed = extract_boxed(text)
        if boxed:
            match = re.search(r'([A-D])', boxed.upper())
            if match:
                return match.group(1)
        match = re.search(r'\(([A-D])\)', text.upper())
        if match:
            return match.group(1)
        match = re.search(r'([A-D])', text.upper())
        return match.group(1) if match else ""

    print(f"  Valid samples: {len(items)}", flush=True)
    return _run_eval_loop(items, model, tokenizer, args, "gpqa",
                          extract_fn=extract_gpqa,
                          match_fn=lambda p, g: p == g,
                          gen_length=128, steps=128, progress_every=50)


# ============================================================
# BBH — hardest subtasks (n~200 total)
# ============================================================

BBH_HARD_SUBTASKS = [
    "dyck_languages",
    "multistep_arithmetic_two",
    "tracking_shuffled_objects_seven_objects",
    "formal_fallacies",
    "logical_deduction_seven_objects",
    "web_of_lies",
]

@torch.no_grad()
def eval_bbh(model, tokenizer, args):
    print(f"\n{'='*60}", flush=True)
    print(f"BBH Hard Subtasks (α={INFERENCE_ALPHA})", flush=True)
    print(f"{'='*60}", flush=True)

    # Load all subtasks and take ~33 from each (6 subtasks × 33 ≈ 200)
    items = []
    subtask_ranges = {}  # track which items belong to which subtask
    for subtask in BBH_HARD_SUBTASKS:
        try:
            ds = load_dataset("lukaemon/bbh", subtask, split="test")
        except Exception as e:
            print(f"  WARNING: Could not load {subtask}: {e}")
            continue

        # Take first 3 as few-shot examples, eval on next ~33
        shots = []
        for i in range(min(3, len(ds))):
            shots.append(f"Q: {ds[i]['input']}\nA: {ds[i]['target']}")
        shot_text = "\n\n".join(shots) + "\n\n" if shots else ""

        start_idx = 3  # skip few-shot examples
        per_subtask = 33
        subtask_start = len(items)

        for j in range(start_idx, min(start_idx + per_subtask, len(ds))):
            global_idx = len(items)
            prompt = f"{shot_text}Q: {ds[j]['input']}\nA:"
            ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                            max_length=512)["input_ids"].to(model.device)
            gold = ds[j]["target"].strip()
            items.append((global_idx, ids, gold, {"subtask": subtask}))

        subtask_ranges[subtask] = (subtask_start, len(items))
        print(f"  {subtask}: {len(items) - subtask_start} samples (3-shot)")

    print(f"  Total samples: {len(items)}", flush=True)

    def match_bbh(pred, gold):
        return pred.strip().lower() == gold.strip().lower()

    result = _run_eval_loop(items, model, tokenizer, args, "bbh",
                            extract_fn=lambda text: text.strip().split("\n")[0].strip(),
                            match_fn=match_bbh,
                            gen_length=128, steps=128, progress_every=30)

    # Per-subtask breakdown
    per_sample = result.get("per_sample", {})
    print(f"\n  BBH by Subtask:")
    print(f"  {'Subtask':<45} | {'N':>4} | {'Base':>8} | {'Router':>8} | {'Delta':>7}")
    print(f"  {'-'*80}")
    for subtask, (s, e) in subtask_ranges.items():
        indices = [str(i) for i in range(s, e)]
        n_sub = 0
        b_c = 0
        r_c = 0
        for si in indices:
            if si in per_sample and "correct_baseline" in per_sample[si]:
                n_sub += 1
                b_c += int(per_sample[si].get("correct_baseline", False))
                r_c += int(per_sample[si].get("correct_router", False))
        if n_sub > 0:
            print(f"  {subtask:<45} | {n_sub:>4} | {b_c/n_sub:>7.0%} | {r_c/n_sub:>7.0%} | {(r_c-b_c)/n_sub:>+6.0%}")

    return result


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Scaled benchmark evaluation for ALA-LLaDA")
    parser.add_argument("--benchmarks", nargs="+",
                        choices=["gsm8k", "math", "arc", "gpqa", "bbh"],
                        default=["gsm8k", "math", "arc", "gpqa", "bbh"],
                        help="Which benchmarks to run")
    parser.add_argument("--checkpoint-dir", default="checkpoints",
                        help="Directory for checkpoint files")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing checkpoints")
    parser.add_argument("--save-every", type=int, default=50,
                        help="Checkpoint every N samples")
    return parser.parse_args()


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('medium')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    model, tokenizer = load_model()

    results = {}
    total_t0 = time.time()

    # Run order: fast first (ARC, GPQA, BBH), then longer (MATH, GSM8K)
    run_order = []
    for b in ["arc", "gpqa", "bbh", "math", "gsm8k"]:
        if b in args.benchmarks:
            run_order.append(b)

    eval_fns = {
        "gsm8k": lambda: eval_gsm8k(model, tokenizer, args),
        "math": lambda: eval_math(model, tokenizer, args),
        "arc": lambda: eval_arc(model, tokenizer, args),
        "gpqa": lambda: eval_gpqa(model, tokenizer, args),
        "bbh": lambda: eval_bbh(model, tokenizer, args),
    }

    for benchmark in run_order:
        t0 = time.time()
        res = eval_fns[benchmark]()
        # Don't save per_sample in summary (too large)
        results[benchmark] = {k: v for k, v in res.items() if k != "per_sample"}
        elapsed = time.time() - t0
        print(f"  {benchmark} completed in {format_time(elapsed)}")

    # Save combined results
    results_path = "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Final summary
    total_elapsed = time.time() - total_t0
    print(f"\n{'='*60}")
    print(f"BENCHMARK SUMMARY (α={INFERENCE_ALPHA}, total: {format_time(total_elapsed)})")
    print(f"{'='*60}")
    print(f"  {'Benchmark':<15} | {'Baseline':>10} | {'Router':>10} | {'Delta':>10} | {'N':>6}")
    print(f"  {'-'*58}")
    for name, res in results.items():
        b_acc = res["Baseline"]["accuracy"]
        r_acc = res["Router"]["accuracy"]
        n = res["Baseline"]["total"]
        print(f"  {name:<15} | {b_acc:>10.1%} | {r_acc:>10.1%} | {r_acc-b_acc:>+10.1%} | {n:>6}")

    print(f"\n  Results saved to {results_path}")
    print(f"  Checkpoints in {args.checkpoint_dir}/")
