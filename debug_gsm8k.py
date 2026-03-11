"""
MATH Difficulty Split Eval (train α=0.1, inference α=0.02)
===========================================================
Runs MATH 200 samples with difficulty breakdown (Level 1-5).
Router trained at α=0.1, evaluated at inference α=0.02.
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

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

INFERENCE_ALPHA = 0.02

# ============================================================
# Answer extraction
# ============================================================

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
            strict=False
        )
        print(f"Router loaded from {weights_path}")
    else:
        print("WARNING: Using random router weights.")
    model.eval()
    return model, tokenizer, device


# ============================================================
# MATH Difficulty Split
# ============================================================

@torch.no_grad()
def eval_math(model, tokenizer, device, n=200):
    print(f"\n{'='*60}")
    print(f"MATH (n={n}) — train α=0.1, inference α={INFERENCE_ALPHA}")
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

    results = {}
    for mode in ["Baseline", "Router"]:
        use_router = (mode == "Router")
        alpha = INFERENCE_ALPHA if use_router else None
        mode_results = {}

        t0 = time.time()
        correct = 0
        for i, (idx, ids, gold, meta) in enumerate(items):
            torch.manual_seed(42)
            out = generate(model, ids, steps=256, gen_length=256,
                           block_length=32,
                           use_router=use_router, temp=0.0, alpha=alpha)
            response = tokenizer.decode(out[0, ids.shape[1]:],
                                        skip_special_tokens=True).strip()
            pred = extract_boxed(response)
            hit = math_match(pred, gold)
            correct += int(hit)
            mode_results[idx] = (pred, hit, meta)

            if (i + 1) % 50 == 0 or i == 0:
                print(f"    {mode} [{i+1}/{len(items)}] "
                      f"acc={correct/(i+1):.1%}", flush=True)

        acc = correct / len(items) if items else 0
        elapsed = time.time() - t0
        print(f"  {mode}: {correct}/{len(items)} ({acc:.1%}) in {elapsed:.0f}s", flush=True)
        results[mode] = mode_results

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
# GSM8K (quick check)
# ============================================================

@torch.no_grad()
def eval_gsm8k(model, tokenizer, device, n=200):
    print(f"\n{'='*60}")
    print(f"GSM8K (n={n}) — inference α={INFERENCE_ALPHA}")
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

    results = {}
    for mode in ["Baseline", "Router"]:
        use_router = (mode == "Router")
        alpha = INFERENCE_ALPHA if use_router else None
        mode_results = {}

        t0 = time.time()
        correct = 0
        for i, (idx, ids, gold, meta) in enumerate(items):
            torch.manual_seed(42)
            out = generate(model, ids, steps=256, gen_length=256,
                           block_length=32,
                           use_router=use_router, temp=0.0, alpha=alpha)
            response = tokenizer.decode(out[0, ids.shape[1]:],
                                        skip_special_tokens=True).strip()
            pred = extract_answer_gsm8k(response)
            hit = (pred == gold)
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
# Main
# ============================================================

if __name__ == "__main__":
    model, tokenizer, device = load_model()

    print(f"\nConfig: train α=0.1, inference α={INFERENCE_ALPHA}")

    # MATH with difficulty breakdown
    math_results = eval_math(model, tokenizer, device, n=200)
    base_c = sum(1 for v in math_results["Baseline"].values() if v[1])
    rout_c = sum(1 for v in math_results["Router"].values() if v[1])
    n = len(math_results["Baseline"])

    # # GSM8K quick check (already have results, skipping for now)
    # gsm8k_results = eval_gsm8k(model, tokenizer, device, n=200)
    # gsm_base = sum(1 for v in gsm8k_results["Baseline"].values() if v[1])
    # gsm_rout = sum(1 for v in gsm8k_results["Router"].values() if v[1])
    # gsm_n = len(gsm8k_results["Baseline"])

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY (train α=0.1, inference α={INFERENCE_ALPHA})")
    print(f"{'='*60}")
    print(f"  MATH: {base_c}/{n} ({base_c/n:.1%}) baseline, {rout_c}/{n} ({rout_c/n:.1%}) router, delta {(rout_c-base_c)/n:+.1%}")

    results = {
        "MATH": {"Baseline": base_c/n, "Router": rout_c/n},
    }
    with open("gate_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to gate_eval_results.json")
