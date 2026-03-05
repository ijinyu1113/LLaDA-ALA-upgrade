"""Quick punctuation test: flat alpha=0.1, 3 samples at temp=0."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
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
else:
    print("WARNING: random router weights")

model.eval()

# Monkey-patch forward to use flat alpha=0.1
_orig_forward = model.forward
def _flat_alpha_forward(input_ids, **kwargs):
    kwargs["alpha"] = 0.1
    return _orig_forward(input_ids, **kwargs)
model.forward = _flat_alpha_forward

prompt = "Write a short story about a cat who finds a magical portal in a library."
ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

print(f"\nPrompt: {prompt}")
print("=" * 60)

for mode in ["Baseline", "Router (flat alpha=0.1)"]:
    use_router = (mode != "Baseline")
    print(f"\n--- {mode} ---")
    torch.manual_seed(42)
    out = generate(model, ids, steps=64, gen_length=64,
                   use_router=use_router, temp=0.0)
    gen_ids = out[0, ids.shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    print(f"  Text: {text}")
    print(f"  Token IDs: {gen_ids.tolist()}")
    # Show each token individually to see spaces
    tokens = [tokenizer.decode([tid]) for tid in gen_ids.tolist()]
    print(f"  Tokens:    {tokens}")
