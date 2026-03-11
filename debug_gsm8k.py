"""
Confidence Gate Evaluation (α=0.05)
====================================
Baseline vs Router across multiple benchmarks.
Tests whether the learned confidence gate prevents regression on easy tasks.

Benchmarks:
  - GSM8K (200 samples, steps=256, gen=256)
  - MATH (200 samples, steps=256, gen=256) + breakdown by difficulty level
  - ARC-Challenge (200 samples, steps=32, gen=32)
  - GPQA Diamond (~198 samples, steps=256, gen=256)
  - BBH hardest subtasks (50 each × 6 subtasks, steps=64, gen=64)
"""
import torch
import re
import sys
import os
import time
import json

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, concatenate_datasets
from generation_utils import generate
from models import ALALLaDA, MASK_TOKEN_ID

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# CUDA optimizations
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ============================================================
# Answer extraction
# ============================================================

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
    match = re.search(r'([A-E])', text.upper())
    return match.group(1) if match else ""

def extract_answer_mc(text):
    """Extract A/B/C/D from multiple-choice response."""
    match = re.search(r'\b([A-D])\b', text.upper())
    return match.group(1) if match else ""

# ============================================================
# Model Loading
# ============================================================

def load_model():
    model_id = "GSAI-ML/LLaDA-8B-Instruct"
    print("Loading model...", flush=True)
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
            strict=False  # allow missing conf_gate from old checkpoints
        )
        print(f"Router loaded from {weights_path}")
    else:
        print("WARNING: Using random router weights.")
    model.eval()
    return model, tokenizer, device


def run_eval(items, model, tokenizer, steps, gen_length, block_length,
             extract_fn, match_fn=None):
    """Run baseline + router (α=0.05) on a list of (idx, prompt_ids, gold, meta) items.

    Returns: {mode: {idx: (pred, correct, meta)}}
    """
    if match_fn is None:
        match_fn = lambda p, g: p == g

    results = {}
    for mode in ["Baseline", "Router"]:
        use_router = (mode == "Router")
        mode_results = {}

        t0 = time.time()
        correct = 0
        for i, (idx, ids, gold, meta) in enumerate(items):
            torch.manual_seed(42)
            out = generate(model, ids, steps=steps, gen_length=gen_length,
                           block_length=block_length,
                           use_router=use_router, temp=0.0)
            response = tokenizer.decode(out[0, ids.shape[1]:],
                                        skip_special_tokens=True).strip()
            pred = extract_fn(response)
            hit = match_fn(pred, gold)
            correct += int(hit)
            mode_results[idx] = (pred, hit, meta)

            if (i + 1) % 50 == 0 or i == 0:
                print(f"    {mode} [{i+1}/{len(items)}] "
                      f"acc={correct/(i+1):.1%}", flush=True)

        acc = correct / len(items) if items else 0
        elapsed = time.time() - t0
        print(f"  {mode}: {correct}/{len(items)} ({acc:.1%}) in {elapsed:.0f}s", flush=True)
        results[mode] = mode_results

    return results


# ============================================================
# Benchmark: GSM8K
# ============================================================

@torch.no_grad()
def eval_gsm8k(model, tokenizer, device, n=200):
    print(f"\n{'='*60}")
    print(f"GSM8K (n={n})")
    print(f"{'='*60}", flush=True)

    ds = load_dataset("openai/gsm8k", "main", split="test")
    items = []
    for idx in range(min(n, len(ds))):
        gold = extract_gold_gsm8k(ds[idx]["answer"])
        if not gold:
            continue
        prompt = (f"Solve this math problem step by step. "
                  f"Give your final answer after ####.\n\n"
                  f"Question: {ds[idx]['question']}\n\nSolution:")
        ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=512)["input_ids"].to(device)
        items.append((idx, ids, gold, {}))

    print(f"  Valid samples: {len(items)}", flush=True)
    return run_eval(items, model, tokenizer,
                    steps=256, gen_length=256, block_length=32,
                    extract_fn=extract_answer_gsm8k)


# ============================================================
# Benchmark: MATH (with difficulty breakdown)
# ============================================================

@torch.no_grad()
def eval_math(model, tokenizer, device, n=200):
    print(f"\n{'='*60}")
    print(f"MATH (n={n}) + Difficulty Breakdown")
    print(f"{'='*60}", flush=True)

    subjects = ["algebra", "counting_and_probability", "geometry",
                "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
    parts = [load_dataset("EleutherAI/hendrycks_math", subj, split="test")
             for subj in subjects]
    ds = concatenate_datasets(parts)

    items = []
    for idx in range(min(n, len(ds))):
        sample = ds[idx]
        gold = extract_boxed(sample["solution"])
        if not gold:
            continue
        level = sample.get("level", "Unknown")
        prompt = (f"Solve this math problem. Put your final answer in \\boxed{{}}.\n\n"
                  f"Problem: {sample['problem']}\n\nSolution:")
        ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=512)["input_ids"].to(device)
        items.append((idx, ids, gold, {"level": level}))

    print(f"  Valid samples: {len(items)}", flush=True)

    def math_match(pred, gold):
        return pred.strip().rstrip('.') == gold.strip().rstrip('.')

    results = run_eval(items, model, tokenizer,
                       steps=256, gen_length=256, block_length=32,
                       extract_fn=extract_boxed, match_fn=math_match)

    # Difficulty breakdown
    print(f"\n  MATH by Difficulty Level:")
    print(f"  {'Level':<10} | {'N':>4} | {'Baseline':>10} | {'Router':>10} | {'Delta':>8}")
    print(f"  {'-'*55}")

    levels = sorted(set(meta["level"] for _, _, _, meta in items))
    for level in levels:
        level_indices = [idx for idx, _, _, meta in items if meta["level"] == level]
        n_level = len(level_indices)
        if n_level == 0:
            continue
        base_c = sum(1 for i in level_indices if results["Baseline"][i][1])
        rout_c = sum(1 for i in level_indices if results["Router"][i][1])
        base_a = base_c / n_level
        rout_a = rout_c / n_level
        print(f"  {level:<10} | {n_level:>4} | {base_c:>3}/{n_level} {base_a:>4.0%} "
              f"| {rout_c:>3}/{n_level} {rout_a:>4.0%} | {rout_a - base_a:>+6.0%}")

    return results


# ============================================================
# Benchmark: ARC-Challenge
# ============================================================

@torch.no_grad()
def eval_arc(model, tokenizer, device, n=200):
    print(f"\n{'='*60}")
    print(f"ARC-Challenge (n={n})")
    print(f"{'='*60}", flush=True)

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    items = []
    for idx in range(min(n, len(ds))):
        sample = ds[idx]
        choices = sample["choices"]
        choice_lines = [f"{l}) {t}" for l, t in zip(choices["label"], choices["text"])]
        prompt = (f"Question: {sample['question']}\n"
                  f"{chr(10).join(choice_lines)}\n\nAnswer:")
        ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=512)["input_ids"].to(device)
        gold = sample["answerKey"].upper()
        items.append((idx, ids, gold, {}))

    print(f"  Valid samples: {len(items)}", flush=True)
    return run_eval(items, model, tokenizer,
                    steps=32, gen_length=32, block_length=32,
                    extract_fn=extract_answer_arc)


# ============================================================
# Benchmark: GPQA Diamond
# ============================================================

@torch.no_grad()
def eval_gpqa(model, tokenizer, device):
    print(f"\n{'='*60}")
    print(f"GPQA Diamond")
    print(f"{'='*60}", flush=True)

    try:
        ds = load_dataset("hendrydong/gpqa_diamond_mc", split="test")
        print(f"  Loaded hendrydong/gpqa_diamond_mc: {len(ds)} samples")
    except Exception as e:
        print(f"  Failed to load GPQA: {e}")
        print("  Skipping GPQA.")
        return None

    items = []
    for idx in range(len(ds)):
        sample = ds[idx]
        problem = sample["problem"]
        gold = extract_boxed(sample["solution"]).strip().upper()
        if not gold:
            continue
        prompt = f"{problem}\n\nAnswer:"
        ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=512)["input_ids"].to(device)
        items.append((idx, ids, gold, {"domain": sample.get("domain", "?")}))

    print(f"  Valid samples: {len(items)}", flush=True)
    return run_eval(items, model, tokenizer,
                    steps=256, gen_length=256, block_length=32,
                    extract_fn=extract_answer_mc)


# ============================================================
# Benchmark: BBH (hardest subtasks)
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
def eval_bbh(model, tokenizer, device, n_per_subtask=50):
    print(f"\n{'='*60}")
    print(f"BBH Hard Subtasks ({n_per_subtask} per subtask)")
    print(f"{'='*60}", flush=True)

    subtask_summaries = {}

    for subtask in BBH_HARD_SUBTASKS:
        print(f"\n  --- {subtask} ---", flush=True)
        try:
            ds = load_dataset("lukaemon/bbh", subtask, split="test")
        except Exception as e:
            print(f"  Failed to load {subtask}: {e}")
            continue

        items = []
        for idx in range(min(n_per_subtask, len(ds))):
            sample = ds[idx]
            prompt = f"{sample['input']}\n\nAnswer:"
            ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                            max_length=512)["input_ids"].to(device)
            gold = sample["target"].strip()
            items.append((idx, ids, gold, {"subtask": subtask}))

        def bbh_match(pred, gold):
            return pred.strip().lower() == gold.strip().lower()

        def bbh_extract(text):
            return text.split('\n')[0].strip()

        results = run_eval(items, model, tokenizer,
                           steps=64, gen_length=64, block_length=32,
                           extract_fn=bbh_extract, match_fn=bbh_match)

        base_c = sum(1 for v in results["Baseline"].values() if v[1])
        rout_c = sum(1 for v in results["Router"].values() if v[1])
        n = len(items)
        subtask_summaries[subtask] = {
            "Baseline": base_c / n if n else 0,
            "Router": rout_c / n if n else 0,
        }

    # Print BBH summary table
    print(f"\n  BBH Summary:")
    print(f"  {'Subtask':<42} | {'Base':>6} | {'Router':>6} | {'Delta':>6}")
    print(f"  {'-'*70}")
    for subtask, s in subtask_summaries.items():
        d = s["Router"] - s["Baseline"]
        print(f"  {subtask:<42} | {s['Baseline']:>5.0%} | {s['Router']:>5.0%} | {d:>+5.0%}")

    return subtask_summaries


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    model, tokenizer, device = load_model()

    all_summaries = {}

    # --- GSM8K ---
    gsm8k_results = eval_gsm8k(model, tokenizer, device, n=200)
    for bench_name, res in [("GSM8K", gsm8k_results)]:
        s = {}
        for mode, mr in res.items():
            c = sum(1 for v in mr.values() if v[1])
            s[mode] = c / len(mr) if mr else 0
        all_summaries[bench_name] = s

    # --- MATH ---
    math_results = eval_math(model, tokenizer, device, n=200)
    s = {}
    for mode, mr in math_results.items():
        c = sum(1 for v in mr.values() if v[1])
        s[mode] = c / len(mr) if mr else 0
    all_summaries["MATH"] = s

    # --- ARC ---
    arc_results = eval_arc(model, tokenizer, device, n=200)
    s = {}
    for mode, mr in arc_results.items():
        c = sum(1 for v in mr.values() if v[1])
        s[mode] = c / len(mr) if mr else 0
    all_summaries["ARC"] = s

    # --- GPQA ---
    gpqa_results = eval_gpqa(model, tokenizer, device)
    if gpqa_results:
        s = {}
        for mode, mr in gpqa_results.items():
            c = sum(1 for v in mr.values() if v[1])
            s[mode] = c / len(mr) if mr else 0
        all_summaries["GPQA"] = s

    # --- BBH ---
    bbh_summaries = eval_bbh(model, tokenizer, device, n_per_subtask=50)
    if bbh_summaries:
        # Aggregate across subtasks
        base_total = sum(s["Baseline"] for s in bbh_summaries.values())
        rout_total = sum(s["Router"] for s in bbh_summaries.values())
        n_sub = len(bbh_summaries)
        all_summaries["BBH (avg)"] = {
            "Baseline": base_total / n_sub if n_sub else 0,
            "Router": rout_total / n_sub if n_sub else 0,
        }

    # ============================================================
    # Grand Summary
    # ============================================================
    print(f"\n{'='*60}")
    print("CONFIDENCE GATE EVAL SUMMARY (α=0.05)")
    print(f"{'='*60}")

    print(f"  {'Benchmark':<12} | {'Baseline':>10} | {'Router':>10} | {'Delta':>10}")
    print(f"  {'-'*50}")

    for bench, s in all_summaries.items():
        base = s.get("Baseline", 0)
        rout = s.get("Router", 0)
        delta = rout - base
        print(f"  {bench:<12} | {base:>9.1%} | {rout:>9.1%} | {delta:>+9.1%}")

    # Save full results
    with open("gate_eval_results.json", "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\n  Results saved to gate_eval_results.json")
