"""Debug GSM8K: print full baseline vs router responses side-by-side.
Runs THREE modes per question: Baseline, Router (with special token protection),
and Router-NoProt (without protection) to test if alpha=0.05 is safe alone."""
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from generation_utils import generate
from models import AMIPRouterInference, ALALLaDA, MASK_TOKEN_ID, SPECIAL_TOKEN_IDS
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

# Monkey-patch to toggle special token protection
_original_forward = ALALLaDA.forward.__wrapped__ if hasattr(ALALLaDA.forward, '__wrapped__') else ALALLaDA.forward

def _forward_no_protect(self, input_ids, attention_mask=None, alpha=None, prompt_length=0):
    """Same as ALALLaDA.forward but skips special token logit restoration."""
    import math
    outputs = self.base_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
    h_L = outputs.hidden_states[-1].to(torch.bfloat16)
    m_idx = [torch.where(row == MASK_TOKEN_ID)[0] for row in input_ids]
    u_idx = [torch.where(row != MASK_TOKEN_ID)[0] for row in input_ids]
    delta = self.router(h_L, m_idx, u_idx)
    if alpha is None:
        from models import ALPHA_BASE, ALPHA_SCALE
        gen_region = input_ids[:, prompt_length:]
        p_mask = (gen_region == MASK_TOKEN_ID).float().mean().item()
        alpha = ALPHA_BASE + ALPHA_SCALE * p_mask
    blended = h_L + alpha * delta
    logits = self.base_model.model.transformer.ff_out(blended)
    if self.base_model.model.config.scale_logits:
        logits *= 1.0 / math.sqrt(self.base_model.model.config.d_model)
    # NO special token restoration here
    return type('Obj', (object,), {'logits': logits})()

print("\n" + "=" * 80)
num_samples = 15
scores = {"Baseline": 0, "Router": 0, "Router-NoProt": 0}

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

    for mode in ["Baseline", "Router", "Router-NoProt"]:
        # Swap forward method for no-protection test
        if mode == "Router-NoProt":
            ALALLaDA.forward = _forward_no_protect
            use_router = True
        elif mode == "Router":
            ALALLaDA.forward = _original_forward
            use_router = True
        else:
            ALALLaDA.forward = _original_forward
            use_router = False

        torch.manual_seed(42)
        out = generate(model, ids, steps=256, gen_length=256,
                       use_router=use_router, temp=0.0)
        response = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        pred = extract_answer(response)
        hit = (pred == gold)
        scores[mode] += int(hit)

        mark = "\u2713" if hit else "\u2717"
        print(f"\n  --- {mode} {mark} (pred={pred}) ---")
        print(f"  {response[:500]}")

        gen_ids = out[0, ids.shape[1]:].tolist()
        last_tokens = [(tid, tokenizer.decode([tid])) for tid in gen_ids[-20:] if tid != 0]
        print(f"  Last tokens: {last_tokens}")

    # Restore original forward
    ALALLaDA.forward = _original_forward

print(f"\n{'='*80}")
print(f"GSM8K Summary (n={num_samples}):")
for mode, correct in scores.items():
    print(f"  {mode:<15} {correct}/{num_samples}")

# ── Logical Reasoning Test ─────────────────────────────────
LOGIC_TEST_CASES = [
    ("Triple Swap",  "Alice has an apple, Bob has a banana, and Charlie has a cherry. Alice swaps with Bob. Then Bob swaps with Charlie. Now, Alice has the", "banana"),
    ("Distractor",   "A gold coin is in the red box. A silver coin is in the blue bag. I replace the gold coin with a copper coin. The red box now has the", "copper"),
    ("Relational",   "The mountain is taller than the hill. The building is shorter than the hill. The shortest object is the", "building"),
    ("State Swap",   "I have a box and a bag. The ball is in the box. The key is in the bag. I swap them. The bag now has the", "ball"),
]

print(f"\n{'='*80}")
print("LOGICAL REASONING (temp=0.0)")
print(f"{'='*80}")
print(f"  {'Category':<15} | {'Expected':<10} | {'Baseline':<20} | {'Router':<20} | {'Router-NoProt':<20}")
print(f"  {'-'*95}")

logic_scores = {"Baseline": 0, "Router": 0, "Router-NoProt": 0}

for category, prompt, expected in LOGIC_TEST_CASES:
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    answers = {}

    for mode in ["Baseline", "Router", "Router-NoProt"]:
        if mode == "Router-NoProt":
            ALALLaDA.forward = _forward_no_protect
            use_router = True
        elif mode == "Router":
            ALALLaDA.forward = _original_forward
            use_router = True
        else:
            ALALLaDA.forward = _original_forward
            use_router = False

        torch.manual_seed(42)
        out = generate(model, prompt_ids, steps=64, gen_length=32,
                       use_router=use_router, temp=0.0)
        ans = tokenizer.decode(out[0, prompt_ids.shape[1]:],
                               skip_special_tokens=True).strip().lower()
        hit = expected.lower() in ans
        logic_scores[mode] += int(hit)
        mark = "\u2713" if hit else "\u2717"
        answers[mode] = f"{mark} {ans[:18]}"

    ALALLaDA.forward = _original_forward
    print(f"  {category:<15} | {expected:<10} | {answers['Baseline']:<20} | "
          f"{answers['Router']:<20} | {answers['Router-NoProt']:<20}")

print(f"\n  Scores: ", end="")
for mode, c in logic_scores.items():
    print(f"{mode} {c}/4  ", end="")
print()

print(f"\n{'='*80}")
print("CONCLUSION:")
print("  If Router-NoProt matches Router on both GSM8K and logical reasoning,")
print("  special token protection is unnecessary at alpha=0.05.")
