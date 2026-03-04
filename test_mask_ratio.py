import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np
import os
import json
import re
import matplotlib.pyplot as plt
from generation_utils import generate, get_num_transfer_tokens, add_gumbel_noise
import math
# ============================================================
# 1. ARCHITECTURE
# ============================================================
class AMIPRouter(torch.nn.Module):
    def __init__(self, d_model=4096, K=8):
        super().__init__()
        self.routing_net = torch.nn.Linear(d_model, K)
        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(d_model * 2, d_model // 4),
                torch.nn.GELU(),
                torch.nn.Linear(d_model // 4, d_model)
            ) for _ in range(K)
        ])
        
    def forward(self, h_L, mask_indices, unmasked_indices, range_r=5):
        delta_h = torch.zeros_like(h_L)
        bsz, seq_len, d_model = h_L.shape

        for b in range(bsz):
            m_idx, u_idx = mask_indices[b], unmasked_indices[b]
            for a in m_idx:
                adj = [t for t in u_idx if 0 < abs(t - a) <= range_r]
                if not adj: continue

                h_mask = h_L[b, a:a+1, :]
                pair_deltas = []
                relevance_scores = []

                for t in adj:
                    h_anchor = h_L[b, t:t+1, :]
                    weights = F.softmax(self.routing_net(h_mask), dim=-1)
                    conditioned_in = torch.cat([h_anchor, h_mask], dim=-1)
                    expert_out = sum(
                        weights[:, i:i+1] * expert(conditioned_in)
                        for i, expert in enumerate(self.experts)
                    )
                    pair_deltas.append(expert_out)
                    score = (h_anchor * h_mask).sum(dim=-1) / (d_model ** 0.5)
                    relevance_scores.append(score)

                scores = torch.cat(relevance_scores, dim=0)
                combine_weights = F.softmax(scores, dim=0)
                stacked = torch.cat(pair_deltas, dim=0)
                weighted_delta = (combine_weights.unsqueeze(-1) * stacked).sum(dim=0)
                delta_h[b, a, :] = weighted_delta

        return delta_h


class ALALLaDA(torch.nn.Module):
    def __init__(self, base_model, alpha=0.1):
        super().__init__()
        self.base_model = base_model
        self.router = AMIPRouter()
        self.alpha = alpha

    @property
    def device(self):
        return next(self.base_model.parameters()).device

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # Last hidden state is POST ln_f, so don't apply ln_f again
        h_L = outputs.hidden_states[-1].to(torch.bfloat16)
        m_idx = [torch.where(row == 126336)[0] for row in input_ids]
        u_idx = [torch.where(row != 126336)[0] for row in input_ids]
        delta = self.router(h_L, m_idx, u_idx)
        blended_h = ((1 - self.alpha) * h_L) + (self.alpha * delta)
        # Skip ln_f — already applied. Go straight to ff_out.
        logits = self.base_model.model.transformer.ff_out(blended_h)
        if self.base_model.model.config.scale_logits:
            logits = logits * (1 / math.sqrt(self.base_model.model.config.d_model))
        return type('Obj', (object,), {'logits': logits})

    def base_logits(self, input_ids):
        return self.base_model(input_ids).logits

# ============================================================
# 3. EVAL: MASK PREDICTION ACCURACY (single-step)
# ============================================================
@torch.no_grad()
def eval_mask_accuracy(model, tokenizer, num_samples=20, p_mask=0.15):
    """
    BERT-style single-step mask prediction accuracy.
    Masks p_mask fraction of tokens, measures argmax accuracy.
    """
    print(f"\n{'='*60}")
    print(f"EVAL 1: Single-Step Mask Prediction Accuracy (p_mask={p_mask})")
    print(f"{'='*60}")

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in dataset["text"][:200] if len(t) > 50][:num_samples]

    correct_base, correct_router, total = 0, 0, 0

    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        ids = enc["input_ids"].to(model.device)
        original = ids.clone()

        mask_prob = torch.full(ids.shape, p_mask, device=ids.device)
        mask_indices = torch.bernoulli(mask_prob).bool()
        mask_indices[:, 0] = False

        masked_ids = ids.clone()
        masked_ids[mask_indices] = 126336

        if not mask_indices.any():
            continue

        b_logits = model.base_logits(masked_ids)
        r_logits = model(masked_ids).logits

        targets = original[mask_indices]
        correct_base += (b_logits.argmax(dim=-1)[mask_indices] == targets).sum().item()
        correct_router += (r_logits.argmax(dim=-1)[mask_indices] == targets).sum().item()
        total += targets.numel()

    base_acc = correct_base / total
    router_acc = correct_router / total
    print(f"  Baseline: {base_acc:.4f}")
    print(f"  Router:   {router_acc:.4f}")
    print(f"  Δ Acc:    {router_acc - base_acc:+.4f}")

    return {"base_acc": base_acc, "router_acc": router_acc}

@torch.no_grad()
def eval_by_mask_ratio(model, tokenizer, num_samples=50):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in dataset["text"][:500] if len(t) > 50][:num_samples]
    
    mask_ratios = [0.15, 0.30, 0.50, 0.70, 0.85, 0.95]
    
    for p_mask in mask_ratios:
        correct_base, correct_router, total = 0, 0, 0
        base_conf_sum, router_conf_sum = 0.0, 0.0
        
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            ids = enc["input_ids"].to(model.device)
            attn_mask = enc["attention_mask"].to(model.device)
            original = ids.clone()
            
            mask_prob = torch.full(ids.shape, p_mask, device=ids.device)
            mask_prob = mask_prob * attn_mask
            mask_prob[:, 0] = 0
            mask_indices = torch.bernoulli(mask_prob).bool()
            
            masked_ids = ids.clone()
            masked_ids[mask_indices] = 126336
            
            if not mask_indices.any():
                continue
            
            b_logits = model.base_logits(masked_ids)
            r_logits = model(masked_ids).logits
            
            targets = original[mask_indices]
            
            # Accuracy
            correct_base += (b_logits.argmax(-1)[mask_indices] == targets).sum().item()
            correct_router += (r_logits.argmax(-1)[mask_indices] == targets).sum().item()
            
            # Confidence on correct token (this is what drives remasking)
            b_probs = F.softmax(b_logits[mask_indices], dim=-1)
            r_probs = F.softmax(r_logits[mask_indices], dim=-1)
            base_conf_sum += b_probs.gather(-1, targets.unsqueeze(-1)).sum().item()
            router_conf_sum += r_probs.gather(-1, targets.unsqueeze(-1)).sum().item()
            
            total += targets.numel()
        
        ba = correct_base / total if total else 0
        ra = correct_router / total if total else 0
        bc = base_conf_sum / total if total else 0
        rc = router_conf_sum / total if total else 0
        
        print(f"  p={p_mask:.2f} | Acc  Base:{ba:.4f} Router:{ra:.4f} Δ:{ra-ba:+.4f} | "
              f"Conf Base:{bc:.4f} Router:{rc:.4f} Δ:{rc-bc:+.4f}")
# ============================================================
# 4. EVAL: LOGICAL REASONING (generative, across temperatures)
# ============================================================
LOGIC_TEST_CASES = [
    ("Triple Swap",  "Alice has an apple, Bob has a banana, and Charlie has a cherry. Alice swaps with Bob. Then Bob swaps with Charlie. Now, Alice has the", "banana"),
    ("Distractor",   "A gold coin is in the red box. A silver coin is in the blue bag. I replace the gold coin with a copper coin. The red box now has the", "copper"),
    ("Relational",   "The mountain is taller than the hill. The building is shorter than the hill. The shortest object is the", "building"),
    ("State Swap",   "I have a box and a bag. The ball is in the box. The key is in the bag. I swap them. The bag now has the", "ball"),
]


@torch.no_grad()
def eval_logical_reasoning(model, tokenizer, temps=[0.0, 0.15, 0.3], gen_length=32, steps=64):
    """
    Generates completions for logical reasoning prompts across multiple temperatures.
    Checks if the expected keyword appears in the generated output.
    Shows how ALA affects accuracy and robustness to temperature.
    """
    print(f"\n{'='*60}")
    print(f"EVAL 2: Logical Reasoning (temps={temps})")
    print(f"{'='*60}")

    results = []

    for temp in temps:
        print(f"\n  --- Temperature: {temp} ---")
        print(f"  {'Category':<15} | {'Expected':<10} | {'Baseline':<30} | {'Router':<30}")
        print(f"  {'-'*90}")

        base_correct, router_correct, total = 0, 0, 0

        for category, prompt, expected in LOGIC_TEST_CASES:
            ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
            torch.manual_seed(42)
            # Baseline
            b_out = generate(model, ids, steps=steps, gen_length=gen_length, use_router=False, temp=temp)
            b_ans = tokenizer.decode(b_out[0, ids.shape[1]:], skip_special_tokens=True).strip().lower()
            torch.manual_seed(42)
            # Router
            r_out = generate(model, ids, steps=steps, gen_length=gen_length, use_router=True, temp=temp)
            r_ans = tokenizer.decode(r_out[0, ids.shape[1]:], skip_special_tokens=True).strip().lower()

            b_hit = expected.lower() in b_ans
            r_hit = expected.lower() in r_ans
            base_correct += int(b_hit)
            router_correct += int(r_hit)
            total += 1

            b_mark = "✓" if b_hit else "✗"
            r_mark = "✓" if r_hit else "✗"
            print(f"  {category:<15} | {expected:<10} | {b_mark} {b_ans[:28]:<28} | {r_mark} {r_ans[:28]:<28}")

        results.append({
            "temp": temp,
            "base_correct": base_correct,
            "router_correct": router_correct,
            "total": total,
            "base_acc": base_correct / total,
            "router_acc": router_correct / total
        })
        print(f"  Score: Baseline {base_correct}/{total} | Router {router_correct}/{total}")

    return results


# ============================================================
# 5. EVAL: DIVERSITY (Jaccard + Unique Token Ratio, across temps)
# ============================================================
def eval_diversity(model, tokenizer, num_samples=5, temps=[0.0, 0.15, 0.3]):
    print(f"\n{'='*60}")
    print(f"EVAL 3: Generation Diversity (temps={temps}, samples={num_samples})")
    print(f"{'='*60}")

    prompt = "Write a short story about a cat who finds a magical portal in a library."

    all_results = []

    with open("generated_stories.txt", "w") as f:
        f.write("=" * 60 + "\n")
        f.write("Generated Short Stories\n")
        f.write("=" * 60 + "\n\n")

        for temp in temps:
            print(f"\n  --- Temperature: {temp} ---")
            f.write(f"--- Temperature: {temp} ---\n\n")
            ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
            temp_results = {"temp": temp}

            for mode in ["Baseline", "Router"]:
                use_router = (mode == "Router")
                texts = []

                for s in range(num_samples):
                    torch.manual_seed(42 + s)
                    torch.cuda.manual_seed(42 + s)
                    out = generate(model, ids, steps=64, gen_length=64, use_router=use_router, temp=temp)
                    text = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
                    texts.append(text)
                    print(f"    [{mode} #{s+1}]: {text[:80]}...")
                    f.write(f"[{mode} #{s+1}]\n{text}\n\n")

                words_list = [t.lower().split() for t in texts]
                flat = [w for wl in words_list for w in wl]
                unique_ratio = len(set(flat)) / len(flat) if flat else 0

                sims = []
                for i in range(len(words_list)):
                    for j in range(i + 1, len(words_list)):
                        s1, s2 = set(words_list[i]), set(words_list[j])
                        if s1 and s2:
                            sims.append(len(s1 & s2) / len(s1 | s2))
                avg_jaccard = np.mean(sims) if sims else 1.0

                temp_results[mode] = {
                    "unique_ratio": unique_ratio,
                    "jaccard": avg_jaccard,
                    "texts": texts
                }
                print(f"    [{mode}] Unique: {unique_ratio:.4f} | Jaccard: {avg_jaccard:.4f}")
                f.write(f"[{mode}] Unique: {unique_ratio:.4f} | Jaccard: {avg_jaccard:.4f}\n\n")

            all_results.append(temp_results)

    print("\n  Stories saved to generated_stories.txt")
    return all_results


# ============================================================
# 6. EVAL: ENTROPY ACROSS DENOISING STEPS
# ============================================================
@torch.no_grad()
def eval_entropy_over_steps(model, tokenizer, num_samples=3, gen_length=128, block_length=32, steps=128):
    """
    Tracks average entropy over masked positions at each denoising step.
    Shows how ALA affects prediction confidence throughout generation.
    """
    print(f"\n{'='*60}")
    print(f"EVAL 4: Entropy Across Denoising Steps")
    print(f"{'='*60}")

    mask_id = 126336
    prompt = "One day, a cat named Whiskers found a magical portal in the library."
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    all_base_ent, all_router_ent = [], []

    for sample in range(num_samples):
        print(f"  Sample {sample+1}/{num_samples}")

        for mode in ["base", "router"]:
            entropies = []
            x = torch.full((1, prompt_ids.shape[1] + gen_length), mask_id, dtype=torch.long, device=model.device)
            x[:, :prompt_ids.shape[1]] = prompt_ids.clone()

            for b_idx in range(num_blocks):
                b_start = prompt_ids.shape[1] + (b_idx * block_length)
                b_end = b_start + block_length
                block_mask = (x[:, b_start:b_end] == mask_id)
                transfer_schedule = get_num_transfer_tokens(block_mask, steps_per_block)

                for i in range(steps_per_block):
                    mask_index = (x == mask_id)
                    logits = model(x).logits if mode == "router" else model.base_logits(x)
                    logits[:, :, 126081] = -torch.inf

                    # Entropy over masked positions
                    masked_logits = logits[mask_index]
                    if masked_logits.shape[0] > 0:
                        probs = F.softmax(masked_logits.float(), dim=-1)
                        ent = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
                    else:
                        ent = 0.0
                    entropies.append(ent)

                    # Unmask step
                    x0 = torch.argmax(logits, dim=-1)
                    probs_conf = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(probs_conf, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                    x0_p[:, b_end:] = -float('inf')
                    x0 = torch.where(mask_index, x0, x)
                    confidence = torch.where(mask_index, x0_p, -float('inf'))

                    transfer_idx = torch.zeros_like(x, dtype=torch.bool)
                    for j in range(confidence.shape[0]):
                        _, sel_idx = torch.topk(confidence[j], k=transfer_schedule[j, i])
                        transfer_idx[j, sel_idx] = True
                    x[transfer_idx] = x0[transfer_idx]

            if mode == "base":
                all_base_ent.append(entropies)
            else:
                all_router_ent.append(entropies)

    # Average across samples
    min_len = min(min(len(e) for e in all_base_ent), min(len(e) for e in all_router_ent))
    avg_base = np.mean([e[:min_len] for e in all_base_ent], axis=0)
    avg_router = np.mean([e[:min_len] for e in all_router_ent], axis=0)
    std_base = np.std([e[:min_len] for e in all_base_ent], axis=0)
    std_router = np.std([e[:min_len] for e in all_router_ent], axis=0)

    print(f"  Mean Base Entropy:   {np.mean(avg_base):.4f}")
    print(f"  Mean Router Entropy: {np.mean(avg_router):.4f}")

    return {
        "avg_base": avg_base, "avg_router": avg_router,
        "std_base": std_base, "std_router": std_router
    }


# ============================================================
# 7. EVAL: FLATNESS TEST
# ============================================================
@torch.no_grad()
def eval_flatness(model, tokenizer):
    """
    Tests whether the router produces a flatter (more uniform) distribution
    over equally valid options. Uses number selection as a proxy.
    Normalized entropy: 1.0 = perfectly uniform, 0.0 = all mass on one token.
    """
    print(f"\n{'='*60}")
    print(f"EVAL 5: Distribution Flatness")
    print(f"{'='*60}")

    prompt = "Choose a number between 1 and 10:"
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    target_tokens = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(1, 11)]

    seq = torch.cat([ids, torch.full((1, 1), 126336, device=model.device)], dim=1)
    results = {}

    for mode in ["Baseline", "Router"]:
        logits = model(seq).logits if mode == "Router" else model.base_logits(seq)
        probs = F.softmax(logits[0, -1, :].float(), dim=-1)
        number_probs = probs[target_tokens]

        ent = -(number_probs * torch.log(number_probs + 1e-9)).sum().item()
        max_ent = np.log(len(target_tokens))
        normalized_ent = ent / max_ent

        results[mode] = {
            "probs": number_probs.detach().float().cpu().numpy(),
            "entropy": ent,
            "normalized_entropy": normalized_ent
        }
        print(f"  [{mode}] Entropy: {ent:.4f} | Normalized: {normalized_ent:.4f}")
        for i in range(len(target_tokens)):
            print(f"    P({i+1}) = {number_probs[i].item():.4f}")

    return results


# ============================================================
# 8. EVAL: GSM8K END-TO-END BENCHMARK
# ============================================================
@torch.no_grad()
def eval_gsm8k(model, tokenizer, num_samples=50, steps=256, gen_length=256, temp=0.0):
    """
    End-to-end accuracy on GSM8K math problems.
    Generates full solutions, extracts final numerical answer, compares to ground truth.
    """
    print(f"\n{'='*60}")
    print(f"EVAL 6: GSM8K End-to-End Accuracy (n={num_samples})")
    print(f"{'='*60}")

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    samples = list(dataset.select(range(min(num_samples, len(dataset)))))

    def extract_answer(text):
        if "####" in text:
            after = text.split("####")[-1].strip()
            match = re.match(r'-?\d+\.?\d*', after)
            if match:
                return match.group()
        numbers = re.findall(r'-?\d+\.?\d*', text)
        return numbers[-1] if numbers else ""

    def extract_gold_answer(answer_text):
        if "####" in answer_text:
            after = answer_text.split("####")[-1].strip()
            return ''.join(c for c in after if c.isdigit() or c == '-')
        return ""

    results = {"Baseline": {"correct": 0, "total": 0}, "Router": {"correct": 0, "total": 0}}

    for idx, sample in enumerate(samples):
        question = sample["question"]
        gold = extract_gold_answer(sample["answer"])
        if not gold:
            continue

        prompt = f"Solve this math problem step by step. Give your final answer after ####.\n\nQuestion: {question}\n\nSolution:"
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(model.device)

        for mode in ["Baseline", "Router"]:
            use_router = (mode == "Router")
            torch.manual_seed(42)
            out = generate(model, ids, steps=steps, gen_length=gen_length, use_router=use_router, temp=temp)
            response = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
            pred = extract_answer(response)

            is_correct = (pred == gold)
            results[mode]["correct"] += int(is_correct)
            results[mode]["total"] += 1

            if idx < 5:
                status = "✓" if is_correct else "✗"
                print(f"  [{mode}] Q{idx+1} {status} | Gold: {gold} | Pred: {pred}")
                if idx < 2:
                    print(f"    Response: {response[:150]}...")

        if (idx + 1) % 10 == 0:
            for mode in ["Baseline", "Router"]:
                acc = results[mode]["correct"] / results[mode]["total"]
                print(f"  Progress {idx+1}/{num_samples} | {mode}: {acc:.4f}")

    for mode in ["Baseline", "Router"]:
        acc = results[mode]["correct"] / results[mode]["total"] if results[mode]["total"] > 0 else 0
        results[mode]["accuracy"] = acc
        print(f"\n  {mode} GSM8K Accuracy: {acc:.4f} ({results[mode]['correct']}/{results[mode]['total']})")

    return results


# ============================================================
# 9. PLOTTING
# ============================================================
def plot_entropy_curve(entropy_results, alpha=0.1, save_path="entropy_curve.png"):
    avg_base = entropy_results["avg_base"]
    avg_router = entropy_results["avg_router"]
    std_base = entropy_results["std_base"]
    std_router = entropy_results["std_router"]

    positions = np.arange(len(avg_base))
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

    ax1 = axes[0]
    ax1.plot(positions, avg_base, color="steelblue", linewidth=1.5, label="Baseline", alpha=0.9)
    ax1.fill_between(positions, avg_base - std_base, avg_base + std_base, color="steelblue", alpha=0.15)
    ax1.plot(positions, avg_router, color="coral", linewidth=1.5, label=f"Router (α={alpha})", alpha=0.9)
    ax1.fill_between(positions, avg_router - std_router, avg_router + std_router, color="coral", alpha=0.15)
    ax1.set_xlabel("Denoising Step")
    ax1.set_ylabel("Avg Entropy (nats)")
    ax1.set_title("Per-Step Entropy During Denoising: Baseline vs ALA Router")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    diff = avg_router - avg_base
    colors = ["coral" if d > 0 else "steelblue" for d in diff]
    ax2.bar(positions, diff, color=colors, alpha=0.7, width=1.0)
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.set_xlabel("Denoising Step")
    ax2.set_ylabel("Δ Entropy")
    ax2.set_title("Entropy Difference (Router - Baseline)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Entropy plot saved to {save_path}")


def plot_flatness(flatness_results, save_path="flatness.png"):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(1, 11)
    width = 0.35

    ax.bar(x - width/2, flatness_results["Baseline"]["probs"], width, label="Baseline", color="steelblue", alpha=0.8)
    ax.bar(x + width/2, flatness_results["Router"]["probs"], width, label="Router", color="coral", alpha=0.8)
    ax.set_xlabel("Number")
    ax.set_ylabel("Probability")
    ax.set_title("Distribution Flatness: P(number) for 'Choose 1-10'")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Flatness plot saved to {save_path}")


def plot_sweep_summary(logic_results, diversity_results, save_path="sweep_summary.png"):
    """Plots accuracy vs diversity across temperatures."""
    temps = [r["temp"] for r in logic_results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Logical accuracy vs temp
    ax1 = axes[0]
    ax1.plot(temps, [r["base_acc"] for r in logic_results], 'o-', color="steelblue", label="Baseline", linewidth=2)
    ax1.plot(temps, [r["router_acc"] for r in logic_results], 's-', color="coral", label="Router", linewidth=2)
    ax1.set_xlabel("Temperature")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Logical Reasoning Accuracy vs Temperature")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)

    # Right: Jaccard (diversity) vs temp
    ax2 = axes[1]
    base_jac = [d["Baseline"]["jaccard"] for d in diversity_results]
    router_jac = [d["Router"]["jaccard"] for d in diversity_results]
    ax2.plot(temps, base_jac, 'o-', color="steelblue", label="Baseline", linewidth=2)
    ax2.plot(temps, router_jac, 's-', color="coral", label="Router", linewidth=2)
    ax2.set_xlabel("Temperature")
    ax2.set_ylabel("Jaccard Similarity (lower = more diverse)")
    ax2.set_title("Generation Diversity vs Temperature")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Sweep summary plot saved to {save_path}")

# ============================================================
# 10. MAIN
# ============================================================
if __name__ == "__main__":
    # --- Load Model ---
    model_id = 'GSAI-ML/LLaDA-8B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Print architecture for verification
    for name, child in base_model.named_children():
        print(f"{name:30s} -> {type(child).__name__}")
    print("\n--- Second level under 'model' ---")
    for name, child in base_model.model.named_children():
        print(f"model.{name:30s} -> {type(child).__name__}")
    print("\n--- Second level under 'model.transformer' ---")
    for name, child in base_model.model.transformer.named_children():
        print(f"model.transformer.{name:30s} -> {type(child).__name__}")

    model = ALALLaDA(base_model, alpha=0.1).to(torch.bfloat16)
    device = next(base_model.parameters()).device
    model.router.to(device)
    print("Base model type:", type(model.base_model))
    
    dummy_ids = tokenizer("Hello", return_tensors="pt")["input_ids"].to(model.device)
    
    base_out = model.base_model(dummy_ids)
    print("Base output type:", type(base_out))
    print("Has .logits:", hasattr(base_out, 'logits'))
    
    router_out = model(dummy_ids)
    print("Router output type:", type(router_out))
    print("Has .logits:", hasattr(router_out, 'logits'))
    
    if hasattr(base_out, 'logits'):
        print("Base logits shape:", base_out.logits.shape)
        print("Router logits shape:", router_out.logits.shape)
        
        # Check if manual base_logits matches direct .logits
        manual = model.base_logits(dummy_ids)
        direct = base_out.logits
        print("Manual vs direct max diff:", (manual - direct).abs().max().item())
    import inspect
    print(inspect.getsource(type(model.base_model.model).forward))

    print("=" * 60)
    weights_path = "amip_router_best.pt"
    if os.path.exists(weights_path):
        model.router.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Router loaded from {weights_path}")
    else:
        print("WARNING: Using random router weights.")

    model.eval()

    # --- Temperature sweep config ---
    sweep_temps = [0.0, 0.15, 0.3]

    # ===========================================================
    # RUN ALL EVALS
    # ===========================================================
    all_results = {}
    all_results["mask_accuracy_by_ratio"] = eval_by_mask_ratio(model, tokenizer, num_samples=50)
    
    # # EVAL 1: Mask Prediction Accuracy (temperature-independent)
    # all_results["mask_accuracy"] = eval_mask_accuracy(model, tokenizer, num_samples=20)
    # # EVAL 2: Logical Reasoning across temperatures
    # logic_results = eval_logical_reasoning(model, tokenizer, temps=sweep_temps)
    # all_results["logical_reasoning"] = logic_results

    # # EVAL 3: Diversity across temperatures
    # diversity_results = eval_diversity(model, tokenizer, num_samples=5, temps=sweep_temps)
    # all_results["diversity"] = [
    #     {
    #         "temp": d["temp"],
    #         "Baseline": {"unique_ratio": d["Baseline"]["unique_ratio"], "jaccard": d["Baseline"]["jaccard"]},
    #         "Router": {"unique_ratio": d["Router"]["unique_ratio"], "jaccard": d["Router"]["jaccard"]}
    #     }
    #     for d in diversity_results
    # ]

    # # EVAL 4: Entropy across denoising steps
    # entropy_res = eval_entropy_over_steps(model, tokenizer, num_samples=3)
    # all_results["entropy"] = {
    #     "mean_base": float(np.mean(entropy_res["avg_base"])),
    #     "mean_router": float(np.mean(entropy_res["avg_router"]))
    # }
    # plot_entropy_curve(entropy_res)

    # # EVAL 5: Flatness
    # flatness_res = eval_flatness(model, tokenizer)
    # all_results["flatness"] = {
    #     mode: {"entropy": r["entropy"], "normalized_entropy": r["normalized_entropy"]}
    #     for mode, r in flatness_res.items()
    # }
    # plot_flatness(flatness_res)

    # # EVAL 6: GSM8K End-to-End Benchmark
    # gsm8k_res = eval_gsm8k(model, tokenizer, num_samples=50)
    # all_results["gsm8k"] = {
    #     mode: {"accuracy": r["accuracy"], "correct": r["correct"], "total": r["total"]}
    #     for mode, r in gsm8k_res.items()
    # }

    # # --- Sweep Summary Plot ---
    # plot_sweep_summary(logic_results, diversity_results)

    # # ===========================================================
    # # FINAL SUMMARY TABLE
    # # ===========================================================
    # print(f"\n{'='*80}")
    # print(f"COMPLETE EVALUATION SUMMARY")
    # print(f"{'='*80}")

    # # 1. Mask accuracy
    # ma = all_results["mask_accuracy"]
    # print(f"\n  1. Mask Prediction Accuracy (single-step, p=0.15)")
    # print(f"     Baseline: {ma['base_acc']:.4f}  |  Router: {ma['router_acc']:.4f}  |  Δ: {ma['router_acc']-ma['base_acc']:+.4f}")

    # # 2. Logical reasoning across temps
    # print(f"\n  2. Logical Reasoning Accuracy")
    # print(f"     {'Temp':<8} | {'Baseline':<10} | {'Router':<10} | {'Δ':<10}")
    # print(f"     {'-'*42}")
    # for r in logic_results:
    #     delta = r['router_acc'] - r['base_acc']
    #     print(f"     {r['temp']:<8} | {r['base_acc']:<10.4f} | {r['router_acc']:<10.4f} | {delta:<+10.4f}")

    # # 3. Diversity across temps
    # print(f"\n  3. Diversity (Jaccard Similarity — lower is more diverse)")
    # print(f"     {'Temp':<8} | {'Base Jaccard':<14} | {'Router Jaccard':<16} | {'Base Unique':<14} | {'Router Unique':<14}")
    # print(f"     {'-'*70}")
    # for d in all_results["diversity"]:
    #     print(f"     {d['temp']:<8} | {d['Baseline']['jaccard']:<14.4f} | {d['Router']['jaccard']:<16.4f} | {d['Baseline']['unique_ratio']:<14.4f} | {d['Router']['unique_ratio']:<14.4f}")

    # # 4. Entropy
    # ent = all_results["entropy"]
    # print(f"\n  4. Entropy Across Denoising Steps")
    # print(f"     Mean Base: {ent['mean_base']:.4f}  |  Mean Router: {ent['mean_router']:.4f}  |  Δ: {ent['mean_router']-ent['mean_base']:+.4f}")

    # # 5. Flatness
    # fl = all_results["flatness"]
    # print(f"\n  5. Distribution Flatness (normalized entropy, 1.0 = uniform)")
    # print(f"     Baseline: {fl['Baseline']['normalized_entropy']:.4f}  |  Router: {fl['Router']['normalized_entropy']:.4f}")

    # # 6. GSM8K
    # gs = all_results["gsm8k"]
    # print(f"\n  6. GSM8K End-to-End Accuracy")
    # print(f"     Baseline: {gs['Baseline']['accuracy']:.4f} ({gs['Baseline']['correct']}/{gs['Baseline']['total']})")
    # print(f"     Router:   {gs['Router']['accuracy']:.4f} ({gs['Router']['correct']}/{gs['Router']['total']})")

    # print(f"\n{'='*80}")

    # # Save all results
    # with open("eval_results.json", "w") as f:
    #     json.dump(all_results, f, indent=2, default=str)
    # print("Results saved to eval_results.json")
    # print("Plots saved: entropy_curve.png, flatness.png, sweep_summary.png")
