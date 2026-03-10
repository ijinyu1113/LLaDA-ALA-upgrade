"""Alpha sweep: test α = 0.01..0.06 on 200 GSM8K + 200 MATH samples."""
import torch
import re
import sys
import os
import time
import math

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

ALPHAS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
NUM_SAMPLES = 200

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

# ============================================================
# Model Loading
# ============================================================

model_id = "GSAI-ML/LLaDA-8B-Instruct"
print("Loading model...")
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
    model.router.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"Router loaded from {weights_path}")
model.eval()

# ============================================================
# GSM8K Alpha Sweep
# ============================================================

print(f"\n{'='*60}")
print(f"GSM8K Alpha Sweep (n={NUM_SAMPLES})")
print(f"{'='*60}", flush=True)

gsm8k = load_dataset("openai/gsm8k", "main", split="test")

# Pre-extract gold answers and prompts
gsm8k_items = []
for idx in range(min(NUM_SAMPLES, len(gsm8k))):
    sample = gsm8k[idx]
    gold = extract_gold_gsm8k(sample["answer"])
    if not gold:
        continue
    prompt = (
        f"Solve this math problem step by step. "
        f"Give your final answer after ####.\n\n"
        f"Question: {sample['question']}\n\nSolution:"
    )
    ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512
                    )["input_ids"].to(device)
    gsm8k_items.append((idx, ids, gold))

print(f"  Valid samples: {len(gsm8k_items)}", flush=True)

# Run baseline once (alpha-independent)
print(f"\n  Running Baseline...", flush=True)
t0 = time.time()
base_correct = 0
base_preds = {}
for i, (idx, ids, gold) in enumerate(gsm8k_items):
    torch.manual_seed(42)
    out = generate(model, ids, steps=256, gen_length=256, use_router=False, temp=0.0)
    response = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
    pred = extract_answer_gsm8k(response)
    hit = (pred == gold)
    base_correct += int(hit)
    base_preds[idx] = pred
    if (i + 1) % 50 == 0:
        print(f"    [{i+1}/{len(gsm8k_items)}] acc={base_correct/(i+1):.1%}", flush=True)

base_acc = base_correct / len(gsm8k_items)
print(f"  Baseline: {base_correct}/{len(gsm8k_items)} ({base_acc:.1%}) in {time.time()-t0:.0f}s", flush=True)

# Run each alpha
gsm8k_results = {"Baseline": base_acc}
for alpha in ALPHAS:
    print(f"\n  Running alpha={alpha}...", flush=True)
    t0 = time.time()
    correct = 0
    for i, (idx, ids, gold) in enumerate(gsm8k_items):
        torch.manual_seed(42)
        out = generate(model, ids, steps=256, gen_length=256,
                       use_router=True, temp=0.0, alpha=alpha)
        response = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        pred = extract_answer_gsm8k(response)
        hit = (pred == gold)
        correct += int(hit)
        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{len(gsm8k_items)}] acc={correct/(i+1):.1%}", flush=True)

    acc = correct / len(gsm8k_items)
    gsm8k_results[f"alpha={alpha}"] = acc
    print(f"  alpha={alpha}: {correct}/{len(gsm8k_items)} ({acc:.1%}) "
          f"delta={acc - base_acc:+.1%} in {time.time()-t0:.0f}s", flush=True)

# ============================================================
# MATH Alpha Sweep
# ============================================================

print(f"\n{'='*60}")
print(f"MATH Alpha Sweep (n={NUM_SAMPLES})")
print(f"{'='*60}", flush=True)

math_subjects = ["algebra", "counting_and_probability", "geometry",
                 "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
math_parts = [load_dataset("EleutherAI/hendrycks_math", subj, split="test")
              for subj in math_subjects]
math_ds = concatenate_datasets(math_parts)

math_items = []
for idx in range(min(NUM_SAMPLES, len(math_ds))):
    sample = math_ds[idx]
    gold = extract_boxed(sample["solution"])
    if not gold:
        continue
    prompt = (
        f"Solve this math problem. Put your final answer in \\boxed{{}}.\n\n"
        f"Problem: {sample['problem']}\n\nSolution:"
    )
    ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512
                    )["input_ids"].to(device)
    math_items.append((idx, ids, gold))

print(f"  Valid samples: {len(math_items)}", flush=True)

# Run baseline once
print(f"\n  Running Baseline...", flush=True)
t0 = time.time()
base_correct = 0
for i, (idx, ids, gold) in enumerate(math_items):
    torch.manual_seed(42)
    out = generate(model, ids, steps=256, gen_length=256, use_router=False, temp=0.0)
    response = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
    pred = extract_boxed(response)
    hit = (pred.strip().rstrip('.') == gold.strip().rstrip('.'))
    base_correct += int(hit)
    if (i + 1) % 50 == 0:
        print(f"    [{i+1}/{len(math_items)}] acc={base_correct/(i+1):.1%}", flush=True)

base_acc = base_correct / len(math_items)
print(f"  Baseline: {base_correct}/{len(math_items)} ({base_acc:.1%}) in {time.time()-t0:.0f}s", flush=True)

# Run each alpha
math_results = {"Baseline": base_acc}
for alpha in ALPHAS:
    print(f"\n  Running alpha={alpha}...", flush=True)
    t0 = time.time()
    correct = 0
    for i, (idx, ids, gold) in enumerate(math_items):
        torch.manual_seed(42)
        out = generate(model, ids, steps=256, gen_length=256,
                       use_router=True, temp=0.0, alpha=alpha)
        response = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        pred = extract_boxed(response)
        hit = (pred.strip().rstrip('.') == gold.strip().rstrip('.'))
        correct += int(hit)
        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{len(math_items)}] acc={correct/(i+1):.1%}", flush=True)

    acc = correct / len(math_items)
    math_results[f"alpha={alpha}"] = acc
    print(f"  alpha={alpha}: {correct}/{len(math_items)} ({acc:.1%}) "
          f"delta={acc - base_acc:+.1%} in {time.time()-t0:.0f}s", flush=True)

# ============================================================
# Summary
# ============================================================

print(f"\n{'='*60}")
print("ALPHA SWEEP SUMMARY")
print(f"{'='*60}")

print(f"\n  {'Alpha':<12} | {'GSM8K':>10} | {'MATH':>10}")
print(f"  {'-'*38}")
print(f"  {'Baseline':<12} | {gsm8k_results['Baseline']:>9.1%} | {math_results['Baseline']:>9.1%}")
for alpha in ALPHAS:
    k = f"alpha={alpha}"
    g = gsm8k_results[k]
    m = math_results[k]
    gd = g - gsm8k_results['Baseline']
    md = m - math_results['Baseline']
    print(f"  {alpha:<12} | {g:>7.1%} ({gd:+.1%}) | {m:>7.1%} ({md:+.1%})")

print(f"\n  Best GSM8K alpha: ", end="")
best_g = max(ALPHAS, key=lambda a: gsm8k_results[f"alpha={a}"])
print(f"{best_g} ({gsm8k_results[f'alpha={best_g}']:.1%})")

print(f"  Best MATH alpha:  ", end="")
best_m = max(ALPHAS, key=lambda a: math_results[f"alpha={a}"])
print(f"{best_m} ({math_results[f'alpha={best_m}']:.1%})")
