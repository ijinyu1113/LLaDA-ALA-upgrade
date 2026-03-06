"""Debug GSM8K: print full baseline vs router responses side-by-side for 10 problems."""
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from generation_utils import generate
from models import AMIPRouterInference, ALALLaDA, MASK_TOKEN_ID
import os

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

dataset = load_dataset("openai/gsm8k", "main", split="test")

def extract_answer(text):
    if "####" in text:
        after = text.split("####")[-1].strip()
        match = re.match(r'-?\d+\.?\d*', after)
        if match:
            return match.group()
    numbers = re.findall(r'-?\d+\.?\d*', text)
    return numbers[-1] if numbers else ""

def extract_gold(answer_text):
    if "####" in answer_text:
        after = answer_text.split("####")[-1].strip()
        return ''.join(c for c in after if c.isdigit() or c == '-')
    return ""

print("\n" + "=" * 80)
num_samples = 15
base_correct, router_correct = 0, 0

for idx in range(num_samples):
    sample = dataset[idx]
    question = sample["question"]
    gold = extract_gold(sample["answer"])
    if not gold:
        continue

    prompt = (
        f"Solve this math problem step by step. "
        f"Give your final answer after ####.\n\n"
        f"Question: {question}\n\nSolution:"
    )
    ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)

    print(f"\n{'='*80}")
    print(f"Q{idx+1}: {question[:120]}")
    print(f"Gold: {gold}")
    print(f"{'='*80}")

    for mode in ["Baseline", "Router"]:
        use_router = (mode == "Router")
        torch.manual_seed(42)
        out = generate(model, ids, steps=256, gen_length=256,
                       use_router=use_router, temp=0.0)
        response = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        pred = extract_answer(response)
        hit = (pred == gold)

        if mode == "Baseline":
            base_correct += int(hit)
        else:
            router_correct += int(hit)

        mark = "✓" if hit else "✗"
        print(f"\n  --- {mode} {mark} (pred={pred}) ---")
        print(f"  {response[:500]}")

        # Show raw token IDs for last 20 tokens (check for weird tokens)
        gen_ids = out[0, ids.shape[1]:].tolist()
        last_tokens = [(tid, tokenizer.decode([tid])) for tid in gen_ids[-20:] if tid != 0]
        print(f"  Last tokens: {last_tokens}")

print(f"\n{'='*80}")
print(f"Summary: Baseline {base_correct}/{num_samples} | Router {router_correct}/{num_samples}")
