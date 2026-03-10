"""
Scaled Benchmark Evaluation for ALA-LLaDA
==========================================
Runs GSM8K (full 1319), MATH (200), and ARC-Challenge (1172) with
checkpointing for crash recovery on long runs.

Usage:
    python run_benchmarks.py --benchmarks gsm8k math arc --resume
    python run_benchmarks.py --benchmarks arc              # quick validation
    python run_benchmarks.py --benchmarks gsm8k --resume   # overnight run
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


# --- Answer extraction (reused from test_router.py) ---

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
            torch.load(weights_path, map_location=device)
        )
        print(f"Router loaded from {weights_path}")
    else:
        print("WARNING: Using random router weights.")

    model.eval()
    print(f"Alpha schedule: alpha = {ALPHA_BASE} + {ALPHA_SCALE} * p_mask (flat at {ALPHA_BASE})")
    return model, tokenizer


# ============================================================
# GSM8K Full (n=1319)
# ============================================================

@torch.no_grad()
def eval_gsm8k_full(model, tokenizer, args):
    print(f"\n{'='*60}", flush=True)
    print("GSM8K Full Test Set (n=1319)", flush=True)
    print(f"{'='*60}", flush=True)

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    num_samples = len(dataset)
    print(f"  Total samples: {num_samples}", flush=True)

    ckpt_path = os.path.join(args.checkpoint_dir, "gsm8k_checkpoint.json")
    ckpt = checkpoint_load(ckpt_path) if args.resume else {}
    completed = set(ckpt.get("completed_indices", []))
    per_sample = ckpt.get("per_sample", {})

    if completed:
        print(f"  Resuming: {len(completed)} samples already done")

    t0 = time.time()
    done_this_run = 0

    for idx in range(num_samples):
        if idx in completed:
            continue

        sample = dataset[idx]
        question = sample["question"]
        gold = extract_gold_gsm8k(sample["answer"])
        if not gold:
            completed.add(idx)
            continue

        prompt = (
            f"Solve this math problem step by step. "
            f"Give your final answer after ####.\n\n"
            f"Question: {question}\n\nSolution:"
        )
        ids = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        )["input_ids"].to(model.device)

        if done_this_run == 0:
            print(f"  First sample: prompt_len={ids.shape[1]}, generating...", flush=True)

        result = {"gold": gold}
        for mode in ["Baseline", "Router"]:
            if done_this_run == 0:
                t_mode = time.time()
                print(f"    {mode} starting...", flush=True)
            use_router = (mode == "Router")
            torch.manual_seed(42)
            out = generate(model, ids, steps=256, gen_length=256,
                           use_router=use_router, temp=0.0)
            response = tokenizer.decode(
                out[0, ids.shape[1]:], skip_special_tokens=True
            ).strip()
            pred = extract_answer_gsm8k(response)
            is_correct = (pred == gold)
            result[f"pred_{mode.lower()}"] = pred
            result[f"correct_{mode.lower()}"] = is_correct
            if done_this_run == 0:
                print(f"    {mode} done in {time.time()-t_mode:.1f}s, pred={pred}, gold={gold}", flush=True)

        per_sample[str(idx)] = result
        completed.add(idx)
        done_this_run += 1

        # Progress reporting
        n_done = len(completed)
        if n_done % 10 == 0 or done_this_run <= 3:
            elapsed = time.time() - t0
            rate = done_this_run / elapsed if elapsed > 0 else 0
            remaining = (num_samples - n_done) / rate if rate > 0 else 0

            base_correct = sum(1 for v in per_sample.values() if v.get("correct_baseline"))
            router_correct = sum(1 for v in per_sample.values() if v.get("correct_router"))
            n_scored = sum(1 for v in per_sample.values() if "correct_baseline" in v)

            print(f"  [{n_done}/{num_samples}] "
                  f"Base: {base_correct}/{n_scored} ({base_correct/n_scored:.1%})  "
                  f"Router: {router_correct}/{n_scored} ({router_correct/n_scored:.1%})  "
                  f"ETA: {format_time(remaining)}")

        # Checkpoint
        if done_this_run % args.save_every == 0:
            checkpoint_save(ckpt_path, {
                "completed_indices": sorted(completed),
                "per_sample": per_sample,
            })

    # Final save
    checkpoint_save(ckpt_path, {
        "completed_indices": sorted(completed),
        "per_sample": per_sample,
    })

    # Summary
    n_scored = sum(1 for v in per_sample.values() if "correct_baseline" in v)
    base_correct = sum(1 for v in per_sample.values() if v.get("correct_baseline"))
    router_correct = sum(1 for v in per_sample.values() if v.get("correct_router"))
    base_acc = base_correct / n_scored if n_scored > 0 else 0
    router_acc = router_correct / n_scored if n_scored > 0 else 0

    print(f"\n  GSM8K Final Results (n={n_scored}):")
    print(f"  Baseline: {base_acc:.4f} ({base_correct}/{n_scored})")
    print(f"  Router:   {router_acc:.4f} ({router_correct}/{n_scored})")
    print(f"  Delta:    {router_acc - base_acc:+.4f}")

    return {
        "Baseline": {"accuracy": base_acc, "correct": base_correct, "total": n_scored},
        "Router": {"accuracy": router_acc, "correct": router_correct, "total": n_scored},
    }


# ============================================================
# MATH Scaled (n=200)
# ============================================================

@torch.no_grad()
def eval_math_scaled(model, tokenizer, args):
    num_samples = 200
    print(f"\n{'='*60}", flush=True)
    print(f"MATH Benchmark (n={num_samples})", flush=True)
    print(f"{'='*60}", flush=True)

    math_subjects = ["algebra", "counting_and_probability", "geometry",
                     "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
    math_parts = [load_dataset("EleutherAI/hendrycks_math", subj, split="test")
                  for subj in math_subjects]
    dataset = concatenate_datasets(math_parts)
    print(f"  Total available: {len(dataset)}, using {num_samples}")

    ckpt_path = os.path.join(args.checkpoint_dir, "math_checkpoint.json")
    ckpt = checkpoint_load(ckpt_path) if args.resume else {}
    completed = set(ckpt.get("completed_indices", []))
    per_sample = ckpt.get("per_sample", {})

    if completed:
        print(f"  Resuming: {len(completed)} samples already done")

    t0 = time.time()
    done_this_run = 0

    for idx in range(num_samples):
        if idx in completed:
            continue

        sample = dataset[idx]
        problem = sample["problem"]
        gold = extract_boxed(sample["solution"])
        if not gold:
            completed.add(idx)
            continue

        prompt = (
            f"Solve this math problem. Put your final answer in \\boxed{{}}.\n\n"
            f"Problem: {problem}\n\nSolution:"
        )
        ids = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        )["input_ids"].to(model.device)

        result = {"gold": gold}
        for mode in ["Baseline", "Router"]:
            use_router = (mode == "Router")
            torch.manual_seed(42)
            out = generate(model, ids, steps=256, gen_length=256,
                           use_router=use_router, temp=0.0)
            response = tokenizer.decode(
                out[0, ids.shape[1]:], skip_special_tokens=True
            ).strip()
            pred = extract_boxed(response)
            pred_norm = pred.strip().rstrip('.')
            gold_norm = gold.strip().rstrip('.')
            is_correct = (pred_norm == gold_norm)
            result[f"pred_{mode.lower()}"] = pred
            result[f"correct_{mode.lower()}"] = is_correct

        per_sample[str(idx)] = result
        completed.add(idx)
        done_this_run += 1

        n_done = len(completed)
        if n_done % 10 == 0 or done_this_run <= 3:
            elapsed = time.time() - t0
            rate = done_this_run / elapsed if elapsed > 0 else 0
            remaining = (num_samples - n_done) / rate if rate > 0 else 0

            base_correct = sum(1 for v in per_sample.values() if v.get("correct_baseline"))
            router_correct = sum(1 for v in per_sample.values() if v.get("correct_router"))
            n_scored = sum(1 for v in per_sample.values() if "correct_baseline" in v)

            print(f"  [{n_done}/{num_samples}] "
                  f"Base: {base_correct}/{n_scored} ({base_correct/n_scored:.1%})  "
                  f"Router: {router_correct}/{n_scored} ({router_correct/n_scored:.1%})  "
                  f"ETA: {format_time(remaining)}")

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

    print(f"\n  MATH Final Results (n={n_scored}):")
    print(f"  Baseline: {base_acc:.4f} ({base_correct}/{n_scored})")
    print(f"  Router:   {router_acc:.4f} ({router_correct}/{n_scored})")
    print(f"  Delta:    {router_acc - base_acc:+.4f}")

    return {
        "Baseline": {"accuracy": base_acc, "correct": base_correct, "total": n_scored},
        "Router": {"accuracy": router_acc, "correct": router_correct, "total": n_scored},
    }


# ============================================================
# ARC-Challenge (n=1172)
# ============================================================

@torch.no_grad()
def eval_arc(model, tokenizer, args):
    print(f"\n{'='*60}", flush=True)
    print("ARC-Challenge (full test set)", flush=True)
    print(f"{'='*60}", flush=True)

    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    num_samples = len(dataset)
    print(f"  Total samples: {num_samples}", flush=True)

    ckpt_path = os.path.join(args.checkpoint_dir, "arc_checkpoint.json")
    ckpt = checkpoint_load(ckpt_path) if args.resume else {}
    completed = set(ckpt.get("completed_indices", []))
    per_sample = ckpt.get("per_sample", {})

    if completed:
        print(f"  Resuming: {len(completed)} samples already done", flush=True)

    t0 = time.time()
    done_this_run = 0

    for idx in range(num_samples):
        if idx in completed:
            continue

        sample = dataset[idx]
        question = sample["question"]
        choices = sample["choices"]
        gold = sample["answerKey"]

        # Build prompt with labeled choices
        choice_lines = []
        for label, text in zip(choices["label"], choices["text"]):
            choice_lines.append(f"{label}) {text}")
        choices_str = "\n".join(choice_lines)

        prompt = (
            f"Question: {question}\n"
            f"{choices_str}\n\n"
            f"Answer:"
        )
        ids = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        )["input_ids"].to(model.device)

        if done_this_run == 0:
            print(f"  First sample: prompt_len={ids.shape[1]}, generating...", flush=True)

        result = {"gold": gold}
        for mode in ["Baseline", "Router"]:
            if done_this_run == 0:
                t_mode = time.time()
                print(f"    {mode} starting...", flush=True)
            use_router = (mode == "Router")
            torch.manual_seed(42)
            out = generate(model, ids, steps=32, gen_length=32,
                           block_length=32, use_router=use_router, temp=0.0)
            response = tokenizer.decode(
                out[0, ids.shape[1]:], skip_special_tokens=True
            ).strip()
            pred = extract_answer_arc(response)
            is_correct = (pred == gold.upper())
            result[f"pred_{mode.lower()}"] = pred
            result[f"correct_{mode.lower()}"] = is_correct
            result[f"response_{mode.lower()}"] = response[:100]
            if done_this_run == 0:
                print(f"    {mode} done in {time.time()-t_mode:.1f}s, pred={pred}, gold={gold}", flush=True)

        per_sample[str(idx)] = result
        completed.add(idx)
        done_this_run += 1

        n_done = len(completed)
        if n_done % 50 == 0 or done_this_run <= 3:
            elapsed = time.time() - t0
            rate = done_this_run / elapsed if elapsed > 0 else 0
            remaining = (num_samples - n_done) / rate if rate > 0 else 0

            base_correct = sum(1 for v in per_sample.values() if v.get("correct_baseline"))
            router_correct = sum(1 for v in per_sample.values() if v.get("correct_router"))
            n_scored = sum(1 for v in per_sample.values() if "correct_baseline" in v)

            print(f"  [{n_done}/{num_samples}] "
                  f"Base: {base_correct}/{n_scored} ({base_correct/n_scored:.1%})  "
                  f"Router: {router_correct}/{n_scored} ({router_correct/n_scored:.1%})  "
                  f"ETA: {format_time(remaining)}")

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

    print(f"\n  ARC-Challenge Final Results (n={n_scored}):")
    print(f"  Baseline: {base_acc:.4f} ({base_correct}/{n_scored})")
    print(f"  Router:   {router_acc:.4f} ({router_correct}/{n_scored})")
    print(f"  Delta:    {router_acc - base_acc:+.4f}")

    return {
        "Baseline": {"accuracy": base_acc, "correct": base_correct, "total": n_scored},
        "Router": {"accuracy": router_acc, "correct": router_correct, "total": n_scored},
    }


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Scaled benchmark evaluation for ALA-LLaDA")
    parser.add_argument("--benchmarks", nargs="+",
                        choices=["gsm8k", "math", "arc"],
                        default=["gsm8k", "math", "arc"],
                        help="Which benchmarks to run")
    parser.add_argument("--checkpoint-dir", default="checkpoints",
                        help="Directory for checkpoint files")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing checkpoints")
    parser.add_argument("--save-every", type=int, default=50,
                        help="Checkpoint every N samples")
    return parser.parse_args()


if __name__ == "__main__":
    # Force unbuffered output for SLURM
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    # CUDA optimizations for GH200
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('medium')  # allow TF32 for matmuls
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    model, tokenizer = load_model()

    results = {}
    total_t0 = time.time()

    # Run in order: ARC (fast) -> MATH (medium) -> GSM8K (long)
    run_order = []
    for b in ["arc", "math", "gsm8k"]:
        if b in args.benchmarks:
            run_order.append(b)

    for benchmark in run_order:
        t0 = time.time()
        if benchmark == "gsm8k":
            results["gsm8k"] = eval_gsm8k_full(model, tokenizer, args)
        elif benchmark == "math":
            results["math"] = eval_math_scaled(model, tokenizer, args)
        elif benchmark == "arc":
            results["arc"] = eval_arc(model, tokenizer, args)
        elapsed = time.time() - t0
        print(f"  {benchmark} completed in {format_time(elapsed)}")

    # Save combined results
    results_path = "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Final summary
    total_elapsed = time.time() - total_t0
    print(f"\n{'='*60}")
    print(f"BENCHMARK SUMMARY (total time: {format_time(total_elapsed)})")
    print(f"{'='*60}")
    print(f"  {'Benchmark':<15} | {'Baseline':>10} | {'Router':>10} | {'Delta':>10} | {'N':>6}")
    print(f"  {'-'*58}")
    for name, res in results.items():
        b_acc = res["Baseline"]["accuracy"]
        r_acc = res["Router"]["accuracy"]
        n = res["Baseline"]["total"]
        print(f"  {name:<15} | {b_acc:>10.4f} | {r_acc:>10.4f} | {r_acc-b_acc:>+10.4f} | {n:>6}")

    print(f"\n  Results saved to {results_path}")
    print(f"  Checkpoints in {args.checkpoint_dir}/")
