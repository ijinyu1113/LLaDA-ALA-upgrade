"""Sweep alpha values on GSM8K to find optimal test-time scaling."""
import torch
import re
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from generation_utils import generate
from models import AMIPRouterInference, ALALLaDA, MASK_TOKEN_ID

# ── Setup ──────────────────────────────────────────────────
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

# ── Dataset ────────────────────────────────────────────────
dataset = load_dataset("openai/gsm8k", "main", split="test")
NUM_SAMPLES = 50

def extract_answer(text):
    if "####" in text:
        after = text.split("####")[-1].strip()
        match = re.match(r'-?\d[\d,]*\.?\d*', after)
        if match:
            return match.group().rstrip('.').replace(',', '')
    numbers = re.findall(r'-?\d[\d,]*\.?\d*', text)
    if numbers:
        return numbers[-1].rstrip('.').replace(',', '')
    return ""

def extract_gold(answer_text):
    if "####" in answer_text:
        after = answer_text.split("####")[-1].strip()
        return ''.join(c for c in after if c.isdigit() or c == '-')
    return ""

# ── Baseline (run once) ───────────────────────────────────
print(f"\nRunning baseline on {NUM_SAMPLES} GSM8K samples...")
base_correct = 0
base_total = 0

for idx in range(NUM_SAMPLES):
    sample = dataset[idx]
    gold = extract_gold(sample["answer"])
    if not gold:
        continue

    prompt = (
        f"Solve this math problem step by step. "
        f"Give your final answer after ####.\n\n"
        f"Question: {sample['question']}\n\nSolution:"
    )
    ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)

    torch.manual_seed(42)
    out = generate(model, ids, steps=256, gen_length=256, use_router=False, temp=0.0)
    response = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
    pred = extract_answer(response)
    base_correct += int(pred == gold)
    base_total += 1

    if (idx + 1) % 10 == 0:
        print(f"  Baseline progress: {idx+1}/{NUM_SAMPLES} | {base_correct}/{base_total}")

base_acc = base_correct / base_total
print(f"Baseline: {base_acc:.4f} ({base_correct}/{base_total})")

# ── Alpha sweep ───────────────────────────────────────────
alphas = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
results = {}

for alpha in alphas:
    print(f"\nSweeping alpha={alpha:.2f}...")
    correct = 0
    total = 0

    for idx in range(NUM_SAMPLES):
        sample = dataset[idx]
        gold = extract_gold(sample["answer"])
        if not gold:
            continue

        prompt = (
            f"Solve this math problem step by step. "
            f"Give your final answer after ####.\n\n"
            f"Question: {sample['question']}\n\nSolution:"
        )
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)

        torch.manual_seed(42)
        out = generate(model, ids, steps=256, gen_length=256, use_router=True, temp=0.0, alpha=alpha)
        response = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        pred = extract_answer(response)
        correct += int(pred == gold)
        total += 1

        if (idx + 1) % 10 == 0:
            print(f"  alpha={alpha:.2f} progress: {idx+1}/{NUM_SAMPLES} | {correct}/{total}")

    acc = correct / total
    results[alpha] = (correct, total, acc)
    print(f"  alpha={alpha:.2f}: {acc:.4f} ({correct}/{total})")

# ── Summary ───────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"ALPHA SWEEP SUMMARY (GSM8K, n={NUM_SAMPLES})")
print(f"{'='*60}")
print(f"  {'Alpha':<8} | {'Accuracy':<10} | {'Correct/Total':<15} | {'vs Baseline'}")
print(f"  {'-'*55}")
print(f"  {'base':<8} | {base_acc:<10.4f} | {base_correct}/{base_total:<13} | ---")
for alpha in alphas:
    c, t, acc = results[alpha]
    delta = acc - base_acc
    sign = "+" if delta >= 0 else ""
    print(f"  {alpha:<8.2f} | {acc:<10.4f} | {c}/{t:<13} | {sign}{delta:.4f}")
print(f"{'='*60}")

best_alpha = max(results, key=lambda a: results[a][2])
print(f"\nBest alpha: {best_alpha} ({results[best_alpha][2]:.4f})")
