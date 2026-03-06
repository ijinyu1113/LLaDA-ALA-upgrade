"""
Unified ALA Evaluation Script
==============================
Runs all evaluations in one script:
  1. Single-step mask prediction accuracy
  2. Mask accuracy & confidence sweep by mask ratio
  3. Logical reasoning (generative, across temperatures)
  4. Generation diversity (Jaccard + unique ratio, across temps)
  5. Entropy across denoising steps
  6. Distribution flatness
  7. GSM8K end-to-end benchmark
  8. MATH end-to-end benchmark

Alpha: flat 0.1 (ALPHA_BASE=0.1, ALPHA_SCALE=0.0)
Residual blending: h + alpha * delta (not interpolation)
Learned Q/K attention aggregation, range_r=10
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np
import os
import json
import re
import matplotlib.pyplot as plt
from generation_utils import generate, get_num_transfer_tokens, add_gumbel_noise
from models import AMIPRouterInference, ALALLaDA, ALPHA_BASE, ALPHA_SCALE, MASK_TOKEN_ID

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True



# ============================================================
# EVAL 1: Single-Step Mask Prediction Accuracy
# ============================================================
@torch.no_grad()
def eval_mask_accuracy(model, tokenizer, num_samples=20, p_mask=0.15):
    """BERT-style single-step mask prediction accuracy at a fixed p_mask."""
    print(f"\n{'='*60}")
    print(f"EVAL 1: Single-Step Mask Prediction Accuracy (p_mask={p_mask})")
    alpha = ALPHA_BASE + ALPHA_SCALE * p_mask
    print(f"  alpha = {alpha:.4f}")
    print(f"{'='*60}")

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in dataset["text"][:200] if len(t) > 50][:num_samples]

    correct_base, correct_router, total = 0, 0, 0

    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        ids = enc["input_ids"].to(model.device)
        original = ids.clone()

        # Create random mask: each position has p_mask probability of being masked
        mask_prob = torch.full(ids.shape, p_mask, device=ids.device)
        mask_indices = torch.bernoulli(mask_prob).bool()  # random True/False per position
        mask_indices[:, 0] = False  # never mask first token (BOS)

        masked_ids = ids.clone()
        masked_ids[mask_indices] = MASK_TOKEN_ID  # replace with [MASK] token

        if not mask_indices.any():
            continue

        b_logits = model.base_logits(masked_ids)          # [B, L, vocab] — baseline predictions
        r_logits = model(masked_ids).logits                # [B, L, vocab] — router predictions (auto-alpha)

        targets = original[mask_indices]                   # [N_masked] — ground-truth tokens at masked positions
        # argmax picks highest-probability token; [mask_indices] selects only masked positions
        correct_base += (b_logits.argmax(dim=-1)[mask_indices] == targets).sum().item()
        correct_router += (r_logits.argmax(dim=-1)[mask_indices] == targets).sum().item()
        total += targets.numel()  # .numel() = number of elements in tensor

    base_acc = correct_base / total
    router_acc = correct_router / total
    print(f"  Baseline: {base_acc:.4f}")
    print(f"  Router:   {router_acc:.4f}")
    print(f"  Δ Acc:    {router_acc - base_acc:+.4f}")

    return {"base_acc": base_acc, "router_acc": router_acc}


# ============================================================
# EVAL 2: Mask Accuracy & Confidence by Mask Ratio
# ============================================================
@torch.no_grad()
def eval_by_mask_ratio(model, tokenizer, num_samples=50):
    """Sweep mask ratios, report accuracy, confidence, and alpha used."""
    print(f"\n{'='*60}")
    print(f"EVAL 2: Accuracy & Confidence by Mask Ratio")
    print(f"{'='*60}")

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in dataset["text"][:500] if len(t) > 50][:num_samples]

    mask_ratios = [0.15, 0.30, 0.50, 0.70, 0.85, 0.95]
    results_list = []

    for p_mask in mask_ratios:
        correct_base, correct_router, total = 0, 0, 0
        base_conf_sum, router_conf_sum = 0.0, 0.0
        alpha = ALPHA_BASE + ALPHA_SCALE * p_mask

        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            ids = enc["input_ids"].to(model.device)
            attn_mask = enc["attention_mask"].to(model.device)
            original = ids.clone()

            mask_prob = torch.full(ids.shape, p_mask, device=ids.device)
            mask_prob = mask_prob * attn_mask  # don't mask padding
            mask_prob[:, 0] = 0               # don't mask BOS
            mask_indices = torch.bernoulli(mask_prob).bool()

            masked_ids = ids.clone()
            masked_ids[mask_indices] = MASK_TOKEN_ID

            if not mask_indices.any():
                continue

            b_logits = model.base_logits(masked_ids)
            r_logits = model(masked_ids).logits  # auto-alpha

            targets = original[mask_indices]

            # Accuracy
            correct_base += (b_logits.argmax(-1)[mask_indices] == targets).sum().item()
            correct_router += (r_logits.argmax(-1)[mask_indices] == targets).sum().item()

            # Confidence = probability assigned to the correct (gold) token
            b_probs = F.softmax(b_logits[mask_indices], dim=-1)  # [N_masked, vocab_size] — probability distributions
            r_probs = F.softmax(r_logits[mask_indices], dim=-1)
            # gather: use targets as column indices to look up P(correct_token) from each distribution
            # targets.unsqueeze(-1) makes [N_masked, 1] to index into [N_masked, vocab] -> [N_masked, 1]
            base_conf_sum += b_probs.gather(-1, targets.unsqueeze(-1)).sum().item()
            router_conf_sum += r_probs.gather(-1, targets.unsqueeze(-1)).sum().item()

            total += targets.numel()

        ba = correct_base / total if total else 0
        ra = correct_router / total if total else 0
        bc = base_conf_sum / total if total else 0
        rc = router_conf_sum / total if total else 0

        results_list.append({
            "p_mask": p_mask, "alpha": alpha,
            "base_acc": ba, "router_acc": ra,
            "base_conf": bc, "router_conf": rc, "n_tokens": total
        })

        print(f"  p={p_mask:.2f} α={alpha:.3f} | "
              f"Acc  Base:{ba:.4f} Router:{ra:.4f} Δ:{ra-ba:+.4f} | "
              f"Conf Base:{bc:.4f} Router:{rc:.4f} Δ:{rc-bc:+.4f}")

    return results_list


# ============================================================
# EVAL 3: Logical Reasoning (generative, across temperatures)
# ============================================================
LOGIC_TEST_CASES = [
    ("Triple Swap",  "Alice has an apple, Bob has a banana, and Charlie has a cherry. Alice swaps with Bob. Then Bob swaps with Charlie. Now, Alice has the", "banana"),
    ("Distractor",   "A gold coin is in the red box. A silver coin is in the blue bag. I replace the gold coin with a copper coin. The red box now has the", "copper"),
    ("Relational",   "The mountain is taller than the hill. The building is shorter than the hill. The shortest object is the", "building"),
    ("State Swap",   "I have a box and a bag. The ball is in the box. The key is in the bag. I swap them. The bag now has the", "ball"),
]


@torch.no_grad()
def eval_logical_reasoning(model, tokenizer, temps=[0.0, 0.15, 0.3],
                           gen_length=32, steps=64):
    """Generate completions for logic prompts; check if expected word appears."""
    print(f"\n{'='*60}")
    print(f"EVAL 3: Logical Reasoning (temps={temps})")
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
            b_out = generate(model, ids, steps=steps, gen_length=gen_length,
                             use_router=False, temp=temp)
            b_ans = tokenizer.decode(b_out[0, ids.shape[1]:],
                                     skip_special_tokens=True).strip().lower()

            torch.manual_seed(42)
            r_out = generate(model, ids, steps=steps, gen_length=gen_length,
                             use_router=True, temp=temp)
            r_ans = tokenizer.decode(r_out[0, ids.shape[1]:],
                                     skip_special_tokens=True).strip().lower()

            b_hit = expected.lower() in b_ans
            r_hit = expected.lower() in r_ans
            base_correct += int(b_hit)
            router_correct += int(r_hit)
            total += 1

            b_mark = "✓" if b_hit else "✗"
            r_mark = "✓" if r_hit else "✗"
            print(f"  {category:<15} | {expected:<10} | "
                  f"{b_mark} {b_ans[:28]:<28} | {r_mark} {r_ans[:28]:<28}")

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
# EVAL 4: Diversity (Jaccard + Unique Token Ratio, across temps)
# ============================================================
@torch.no_grad()
def eval_diversity(model, tokenizer, num_samples=5, temps=[0.0, 0.15, 0.3]):
    """
    At temp=0 all samples are identical (deterministic argmax — not a bug).
    At temp>0 different seeds produce different Gumbel noise -> different outputs.
    """
    print(f"\n{'='*60}")
    print(f"EVAL 4: Generation Diversity (temps={temps}, samples={num_samples})")
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
                    # Each sample gets a distinct seed; generate() no longer
                    # overrides it, so different seeds -> different outputs
                    torch.manual_seed(42 + s)
                    out = generate(model, ids, steps=64, gen_length=64,
                                   use_router=use_router, temp=temp)
                    text = tokenizer.decode(out[0, ids.shape[1]:],
                                            skip_special_tokens=True).strip()
                    texts.append(text)
                    print(f"    [{mode} #{s+1}]: {text[:80]}...")
                    f.write(f"[{mode} #{s+1}]\n{text}\n\n")

                # --- Diversity metrics ---
                words_list = [t.lower().split() for t in texts]  # tokenize each generation into word lists
                flat = [w for wl in words_list for w in wl]      # flatten all words into one list
                unique_ratio = len(set(flat)) / len(flat) if flat else 0  # unique words / total words

                # Jaccard similarity: measure overlap between every pair of generations
                # Lower Jaccard = more diverse outputs
                sims = []
                for i in range(len(words_list)):
                    for j in range(i + 1, len(words_list)):
                        s1, s2 = set(words_list[i]), set(words_list[j])
                        if s1 and s2:
                            # Jaccard = |intersection| / |union| — 1.0 means identical, 0.0 means no overlap
                            sims.append(len(s1 & s2) / len(s1 | s2))
                avg_jaccard = np.mean(sims) if sims else 1.0

                temp_results[mode] = {
                    "unique_ratio": unique_ratio,
                    "jaccard": avg_jaccard,
                    "texts": texts
                }
                print(f"    [{mode}] Unique: {unique_ratio:.4f} | "
                      f"Jaccard: {avg_jaccard:.4f}")
                f.write(f"[{mode}] Unique: {unique_ratio:.4f} | "
                        f"Jaccard: {avg_jaccard:.4f}\n\n")

            all_results.append(temp_results)

    print("\n  Stories saved to generated_stories.txt")
    return all_results


# ============================================================
# EVAL 5: Entropy Across Denoising Steps
# ============================================================
@torch.no_grad()
def eval_entropy_over_steps(model, tokenizer, num_samples=3,
                            gen_length=128, block_length=32, steps=128):
    """
    Track avg entropy over masked positions at each denoising step.
    model(x).logits auto-computes adaptive alpha from the current mask ratio
    of x, so alpha naturally decreases as tokens get committed.
    """
    print(f"\n{'='*60}")
    print(f"EVAL 5: Entropy Across Denoising Steps")
    print(f"{'='*60}")

    mask_id = MASK_TOKEN_ID
    prompt = "One day, a cat named Whiskers found a magical portal in the library."
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    all_base_ent, all_router_ent = [], []

    for sample in range(num_samples):
        print(f"  Sample {sample+1}/{num_samples}")

        for mode in ["base", "router"]:
            entropies = []
            # Initialize: [prompt tokens] + [all MASK tokens for generation region]
            x = torch.full(
                (1, prompt_ids.shape[1] + gen_length),
                mask_id, dtype=torch.long, device=model.device
            )
            x[:, :prompt_ids.shape[1]] = prompt_ids.clone()  # fill in prompt

            # --- Iterative denoising: unmask tokens block-by-block, step-by-step ---
            for b_idx in range(num_blocks):
                b_start = prompt_ids.shape[1] + (b_idx * block_length)
                b_end = b_start + block_length
                block_mask = (x[:, b_start:b_end] == mask_id)  # which positions in this block are still masked
                # Schedule: how many tokens to unmask at each step (e.g., [4, 4, 3, 3, ...])
                transfer_schedule = get_num_transfer_tokens(
                    block_mask, steps_per_block
                )

                for i in range(steps_per_block):  # each step unmasks a few more tokens
                    mask_index = (x == mask_id)    # [B, L] — current mask positions across entire sequence
                    logits = (
                        model(x, prompt_length=prompt_ids.shape[1]).logits
                        if mode == "router" else model.base_logits(x)
                    )
                    logits[:, :, 126081] = -torch.inf  # suppress a special token

                    # --- Measure entropy at masked positions ---
                    # High entropy = model is uncertain; low entropy = model is confident
                    masked_logits = logits[mask_index]  # [N_still_masked, vocab_size]
                    if masked_logits.shape[0] > 0:
                        probs = F.softmax(masked_logits.float(), dim=-1)
                        # Shannon entropy: H = -sum(p * log(p)) — higher = more uncertain
                        ent = -(probs * torch.log(probs + 1e-10)).sum(
                            dim=-1          # sum over vocab dimension -> [N_still_masked]
                        ).mean().item()     # average across all masked positions -> scalar
                    else:
                        ent = 0.0
                    entropies.append(ent)

                    # --- Unmask step: commit the most confident predictions ---
                    x0 = torch.argmax(logits, dim=-1)    # [B, L] — greedy prediction at every position
                    probs_conf = F.softmax(logits, dim=-1)
                    # gather: for each position, get the probability of the predicted token
                    # x0.unsqueeze(-1) makes [B, L, 1] to index into [B, L, vocab] -> [B, L, 1]
                    # squeeze(-1) removes the last dim -> [B, L]
                    x0_p = torch.gather(
                        probs_conf, dim=-1, index=x0.unsqueeze(-1)
                    ).squeeze(-1)                         # [B, L] — confidence per position
                    x0_p[:, b_end:] = -float('inf')       # don't unmask future blocks yet

                    # Only use predictions at masked positions (keep already-committed tokens unchanged)
                    x0 = torch.where(mask_index, x0, x)   # [B, L] — predictions at mask, original elsewhere
                    confidence = torch.where(mask_index, x0_p, -float('inf'))  # -inf for non-masked (already done)

                    # Select the top-k most confident positions to unmask this step
                    transfer_idx = torch.zeros_like(x, dtype=torch.bool)
                    for j in range(confidence.shape[0]):    # loop over batch
                        # topk: returns (values, indices) of k highest elements
                        _, sel_idx = torch.topk(
                            confidence[j], k=transfer_schedule[j, i]  # k = how many to unmask this step
                        )
                        transfer_idx[j, sel_idx] = True
                    x[transfer_idx] = x0[transfer_idx]     # commit: replace [MASK] with predicted token

            if mode == "base":
                all_base_ent.append(entropies)
            else:
                all_router_ent.append(entropies)

    # Average across samples
    min_len = min(
        min(len(e) for e in all_base_ent),
        min(len(e) for e in all_router_ent)
    )
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
# EVAL 6: Distribution Flatness
# ============================================================
@torch.no_grad()
def eval_flatness(model, tokenizer):
    """
    Single mask token after a prompt -> alpha auto-computed from the
    very low mask ratio (≈ 1/seq_len), matching training conditions.
    """
    print(f"\n{'='*60}")
    print(f"EVAL 6: Distribution Flatness")
    print(f"{'='*60}")

    prompt = "Choose a number between 1 and 10:"
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    target_tokens = [
        tokenizer.encode(str(i), add_special_tokens=False)[0]
        for i in range(1, 11)
    ]

    # Append one [MASK] token after the prompt — model predicts what comes next
    seq = torch.cat(
        [ids, torch.full((1, 1), MASK_TOKEN_ID, device=model.device)], dim=1
    )
    results = {}

    for mode in ["Baseline", "Router"]:
        logits = (model(seq, prompt_length=ids.shape[1]).logits
                  if mode == "Router" else model.base_logits(seq))
        probs = F.softmax(logits[0, -1, :].float(), dim=-1)  # probability distribution at the [MASK] position
        number_probs = probs[target_tokens]  # select only P(1), P(2), ..., P(10) using their token IDs

        # Shannon entropy over just the 10 number tokens
        ent = -(number_probs * torch.log(number_probs + 1e-9)).sum().item()
        max_ent = np.log(len(target_tokens))  # max possible entropy = log(10) for uniform distribution
        normalized_ent = ent / max_ent         # 1.0 = perfectly uniform, 0.0 = all probability on one number

        results[mode] = {
            "probs": number_probs.detach().float().cpu().numpy(),
            "entropy": ent,
            "normalized_entropy": normalized_ent
        }
        print(f"  [{mode}] Entropy: {ent:.4f} | "
              f"Normalized: {normalized_ent:.4f}")
        for i in range(len(target_tokens)):
            print(f"    P({i+1}) = {number_probs[i].item():.4f}")

    return results


# ============================================================
# EVAL 7: GSM8K End-to-End Benchmark
# ============================================================
@torch.no_grad()
def eval_gsm8k(model, tokenizer, num_samples=50,
               steps=256, gen_length=256, temp=0.0):
    """End-to-end accuracy on GSM8K math problems."""
    print(f"\n{'='*60}")
    print(f"EVAL 7: GSM8K End-to-End Accuracy (n={num_samples})")
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

    results = {
        "Baseline": {"correct": 0, "total": 0},
        "Router": {"correct": 0, "total": 0}
    }

    for idx, sample in enumerate(samples):
        question = sample["question"]
        gold = extract_gold_answer(sample["answer"])
        if not gold:
            continue

        prompt = (
            f"Solve this math problem step by step. "
            f"Give your final answer after ####.\n\n"
            f"Question: {question}\n\nSolution:"
        )
        ids = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        )["input_ids"].to(model.device)

        for mode in ["Baseline", "Router"]:
            use_router = (mode == "Router")
            torch.manual_seed(42)
            out = generate(model, ids, steps=steps, gen_length=gen_length,
                           use_router=use_router, temp=temp)
            response = tokenizer.decode(
                out[0, ids.shape[1]:], skip_special_tokens=True
            ).strip()
            pred = extract_answer(response)

            is_correct = (pred == gold)
            results[mode]["correct"] += int(is_correct)
            results[mode]["total"] += 1

            if idx < 5:
                status = "✓" if is_correct else "✗"
                print(f"  [{mode}] Q{idx+1} {status} | "
                      f"Gold: {gold} | Pred: {pred}")
                if idx < 2:
                    print(f"    Response: {response[:150]}...")

        if (idx + 1) % 10 == 0:
            for mode in ["Baseline", "Router"]:
                acc = results[mode]["correct"] / results[mode]["total"]
                print(f"  Progress {idx+1}/{num_samples} | {mode}: {acc:.4f}")

    for mode in ["Baseline", "Router"]:
        t = results[mode]["total"]
        acc = results[mode]["correct"] / t if t > 0 else 0
        results[mode]["accuracy"] = acc
        print(f"\n  {mode} GSM8K Accuracy: {acc:.4f} "
              f"({results[mode]['correct']}/{t})")

    return results


# ============================================================
# EVAL 8: MATH End-to-End Benchmark
# ============================================================
@torch.no_grad()
def eval_math(model, tokenizer, num_samples=50,
              steps=256, gen_length=256, temp=0.0):
    """End-to-end accuracy on MATH (competition math) test set."""
    print(f"\n{'='*60}")
    print(f"EVAL 8: MATH End-to-End Accuracy (n={num_samples})")
    print(f"{'='*60}")

    math_subjects = ["algebra", "counting_and_probability", "geometry",
                     "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
    from datasets import concatenate_datasets
    math_parts = [load_dataset("EleutherAI/hendrycks_math", subj, split="test") for subj in math_subjects]
    dataset = concatenate_datasets(math_parts)
    samples = list(dataset.select(range(min(num_samples, len(dataset)))))

    def extract_boxed(text):
        """Extract answer from \\boxed{...} format used in MATH."""
        # Find the last \boxed{...}
        idx = text.rfind("\\boxed{")
        if idx == -1:
            # Try without backslash (model might not produce it)
            idx = text.rfind("boxed{")
            if idx == -1:
                return text.strip().split()[-1] if text.strip() else ""
            idx += 6
        else:
            idx += 7
        # Find matching closing brace
        depth = 1
        end = idx
        while end < len(text) and depth > 0:
            if text[end] == '{':
                depth += 1
            elif text[end] == '}':
                depth -= 1
            end += 1
        return text[idx:end-1].strip() if depth == 0 else ""

    results = {
        "Baseline": {"correct": 0, "total": 0},
        "Router": {"correct": 0, "total": 0}
    }

    for idx, sample in enumerate(samples):
        problem = sample["problem"]
        gold = extract_boxed(sample["solution"])
        if not gold:
            continue

        prompt = (
            f"Solve this math problem. Put your final answer in \\boxed{{}}.\n\n"
            f"Problem: {problem}\n\nSolution:"
        )
        ids = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        )["input_ids"].to(model.device)

        for mode in ["Baseline", "Router"]:
            use_router = (mode == "Router")
            torch.manual_seed(42)
            out = generate(model, ids, steps=steps, gen_length=gen_length,
                           use_router=use_router, temp=temp)
            response = tokenizer.decode(
                out[0, ids.shape[1]:], skip_special_tokens=True
            ).strip()
            pred = extract_boxed(response)

            # Normalize for comparison (strip whitespace, remove trailing periods)
            pred_norm = pred.strip().rstrip('.')
            gold_norm = gold.strip().rstrip('.')
            is_correct = (pred_norm == gold_norm)
            results[mode]["correct"] += int(is_correct)
            results[mode]["total"] += 1

            if idx < 3:
                status = "✓" if is_correct else "✗"
                print(f"  [{mode}] Q{idx+1} {status} | "
                      f"Gold: {gold} | Pred: {pred}")
                if idx < 2:
                    print(f"    Response: {response[:150]}...")

        if (idx + 1) % 10 == 0:
            for mode in ["Baseline", "Router"]:
                acc = results[mode]["correct"] / results[mode]["total"]
                print(f"  Progress {idx+1}/{num_samples} | {mode}: {acc:.4f}")

    for mode in ["Baseline", "Router"]:
        t = results[mode]["total"]
        acc = results[mode]["correct"] / t if t > 0 else 0
        results[mode]["accuracy"] = acc
        print(f"\n  {mode} MATH Accuracy: {acc:.4f} "
              f"({results[mode]['correct']}/{t})")

    return results


# ============================================================
# PLOTTING
# ============================================================
def plot_entropy_curve(entropy_results, save_path="entropy_curve.png"):
    avg_base = entropy_results["avg_base"]
    avg_router = entropy_results["avg_router"]
    std_base = entropy_results["std_base"]
    std_router = entropy_results["std_router"]

    positions = np.arange(len(avg_base))
    fig, axes = plt.subplots(
        2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]}
    )

    ax1 = axes[0]
    ax1.plot(positions, avg_base, color="steelblue", linewidth=1.5,
             label="Baseline", alpha=0.9)
    ax1.fill_between(positions, avg_base - std_base, avg_base + std_base,
                     color="steelblue", alpha=0.15)
    ax1.plot(positions, avg_router, color="coral", linewidth=1.5,
             label="Router (adaptive α)", alpha=0.9)
    ax1.fill_between(positions, avg_router - std_router,
                     avg_router + std_router, color="coral", alpha=0.15)
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

    ax.bar(x - width/2, flatness_results["Baseline"]["probs"], width,
           label="Baseline", color="steelblue", alpha=0.8)
    ax.bar(x + width/2, flatness_results["Router"]["probs"], width,
           label="Router", color="coral", alpha=0.8)
    ax.set_xlabel("Number")
    ax.set_ylabel("Probability")
    ax.set_title("Distribution Flatness: P(number) for 'Choose 1-10'")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Flatness plot saved to {save_path}")


def plot_sweep_summary(logic_results, diversity_results,
                       save_path="sweep_summary.png"):
    """Accuracy vs diversity across temperatures."""
    temps = [r["temp"] for r in logic_results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.plot(temps, [r["base_acc"] for r in logic_results],
             'o-', color="steelblue", label="Baseline", linewidth=2)
    ax1.plot(temps, [r["router_acc"] for r in logic_results],
             's-', color="coral", label="Router", linewidth=2)
    ax1.set_xlabel("Temperature")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Logical Reasoning Accuracy vs Temperature")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)

    ax2 = axes[1]
    base_jac = [d["Baseline"]["jaccard"] for d in diversity_results]
    router_jac = [d["Router"]["jaccard"] for d in diversity_results]
    ax2.plot(temps, base_jac, 'o-', color="steelblue",
             label="Baseline", linewidth=2)
    ax2.plot(temps, router_jac, 's-', color="coral",
             label="Router", linewidth=2)
    ax2.set_xlabel("Temperature")
    ax2.set_ylabel("Jaccard Similarity (lower = more diverse)")
    ax2.set_title("Generation Diversity vs Temperature")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Sweep summary plot saved to {save_path}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # --- Load Model ---
    model_id = 'GSAI-ML/LLaDA-8B-Instruct'
    print("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Wrap base model + inference router into ALALLaDA (uses AMIPRouterInference internally)
    model = ALALLaDA(base_model).to(torch.bfloat16)
    device = next(base_model.parameters()).device  # get device from base model (could be multi-GPU)
    model.router.to(device)

    # Load trained router weights (saved by train_router.py)
    weights_path = "amip_router_best.pt"
    if os.path.exists(weights_path):
        # load_state_dict: loads {param_name: tensor} dict into the router's parameters
        # map_location ensures weights are loaded to the correct device
        model.router.load_state_dict(
            torch.load(weights_path, map_location=device)
        )
        print(f"Router loaded from {weights_path}")
    else:
        print("WARNING: Using random router weights.")

    model.eval()

    print(f"\nAlpha schedule: alpha = {ALPHA_BASE} + {ALPHA_SCALE} * p_mask")
    print(f"  p=0.15 -> alpha={ALPHA_BASE + ALPHA_SCALE*0.15:.4f}")
    print(f"  p=0.50 -> alpha={ALPHA_BASE + ALPHA_SCALE*0.50:.4f}")
    print(f"  p=0.95 -> alpha={ALPHA_BASE + ALPHA_SCALE*0.95:.4f}")

    # ===========================================================
    # RUN ALL EVALS
    # ===========================================================
    all_results = {}
    sweep_temps = [0.0, 0.15, 0.3]

    # EVAL 1: Single-step mask accuracy
    all_results["mask_accuracy"] = eval_mask_accuracy(
        model, tokenizer, num_samples=20
    )

    # EVAL 2: Mask accuracy by ratio sweep
    mask_ratio_results = eval_by_mask_ratio(
        model, tokenizer, num_samples=50
    )
    all_results["mask_accuracy_by_ratio"] = mask_ratio_results

    # EVAL 3: Logical reasoning across temperatures
    logic_results = eval_logical_reasoning(
        model, tokenizer, temps=sweep_temps
    )
    all_results["logical_reasoning"] = logic_results

    # EVAL 4: Diversity across temperatures
    diversity_results = eval_diversity(
        model, tokenizer, num_samples=5, temps=sweep_temps
    )
    all_results["diversity"] = [
        {
            "temp": d["temp"],
            "Baseline": {
                "unique_ratio": d["Baseline"]["unique_ratio"],
                "jaccard": d["Baseline"]["jaccard"]
            },
            "Router": {
                "unique_ratio": d["Router"]["unique_ratio"],
                "jaccard": d["Router"]["jaccard"]
            }
        }
        for d in diversity_results
    ]

    # EVAL 5: Entropy across denoising steps
    entropy_res = eval_entropy_over_steps(model, tokenizer, num_samples=3)
    all_results["entropy"] = {
        "mean_base": float(np.mean(entropy_res["avg_base"])),
        "mean_router": float(np.mean(entropy_res["avg_router"]))
    }
    plot_entropy_curve(entropy_res)

    # EVAL 6: Flatness
    flatness_res = eval_flatness(model, tokenizer)
    all_results["flatness"] = {
        mode: {
            "entropy": r["entropy"],
            "normalized_entropy": r["normalized_entropy"]
        }
        for mode, r in flatness_res.items()
    }
    plot_flatness(flatness_res)

    # EVAL 7: GSM8K End-to-End Benchmark
    gsm8k_res = eval_gsm8k(model, tokenizer, num_samples=100)
    all_results["gsm8k"] = {
        mode: {
            "accuracy": r["accuracy"],
            "correct": r["correct"],
            "total": r["total"]
        }
        for mode, r in gsm8k_res.items()
    }

    # EVAL 8: MATH End-to-End Benchmark
    math_res = eval_math(model, tokenizer, num_samples=50)
    all_results["math"] = {
        mode: {
            "accuracy": r["accuracy"],
            "correct": r["correct"],
            "total": r["total"]
        }
        for mode, r in math_res.items()
    }

    # --- Plots ---
    plot_sweep_summary(logic_results, diversity_results)

    # ===========================================================
    # FINAL SUMMARY TABLE
    # ===========================================================
    print(f"\n{'='*80}")
    print(f"COMPLETE EVALUATION SUMMARY")
    print(f"  Alpha schedule: {ALPHA_BASE} + {ALPHA_SCALE} * p_mask")
    print(f"{'='*80}")

    # 1. Mask accuracy
    ma = all_results["mask_accuracy"]
    print(f"\n  1. Mask Prediction Accuracy (single-step, p=0.15)")
    print(f"     Baseline: {ma['base_acc']:.4f}  |  "
          f"Router: {ma['router_acc']:.4f}  |  "
          f"Δ: {ma['router_acc']-ma['base_acc']:+.4f}")

    # 2. Mask accuracy by ratio
    print(f"\n  2. Mask Accuracy by Ratio")
    print(f"     {'p_mask':<8} | {'α':<6} | {'Base Acc':<10} | "
          f"{'Router Acc':<12} | {'Δ Acc':<10} | {'Δ Conf':<10}")
    print(f"     {'-'*62}")
    for r in mask_ratio_results:
        d_acc = r['router_acc'] - r['base_acc']
        d_conf = r['router_conf'] - r['base_conf']
        print(f"     {r['p_mask']:<8.2f} | {r['alpha']:<6.3f} | "
              f"{r['base_acc']:<10.4f} | {r['router_acc']:<12.4f} | "
              f"{d_acc:<+10.4f} | {d_conf:<+10.4f}")

    # 3. Logical reasoning across temps
    print(f"\n  3. Logical Reasoning Accuracy")
    print(f"     {'Temp':<8} | {'Baseline':<10} | {'Router':<10} | {'Δ':<10}")
    print(f"     {'-'*42}")
    for r in logic_results:
        delta = r['router_acc'] - r['base_acc']
        print(f"     {r['temp']:<8} | {r['base_acc']:<10.4f} | "
              f"{r['router_acc']:<10.4f} | {delta:<+10.4f}")

    # 4. Diversity across temps
    print(f"\n  4. Diversity (Jaccard Similarity — lower is more diverse)")
    print(f"     {'Temp':<8} | {'Base Jaccard':<14} | {'Router Jaccard':<16} "
          f"| {'Base Unique':<14} | {'Router Unique':<14}")
    print(f"     {'-'*70}")
    for d in all_results["diversity"]:
        print(f"     {d['temp']:<8} | "
              f"{d['Baseline']['jaccard']:<14.4f} | "
              f"{d['Router']['jaccard']:<16.4f} | "
              f"{d['Baseline']['unique_ratio']:<14.4f} | "
              f"{d['Router']['unique_ratio']:<14.4f}")

    # 5. Entropy
    ent = all_results["entropy"]
    print(f"\n  5. Entropy Across Denoising Steps")
    print(f"     Mean Base: {ent['mean_base']:.4f}  |  "
          f"Mean Router: {ent['mean_router']:.4f}  |  "
          f"Δ: {ent['mean_router']-ent['mean_base']:+.4f}")

    # 6. Flatness
    fl = all_results["flatness"]
    print(f"\n  6. Distribution Flatness (normalized entropy, 1.0 = uniform)")
    print(f"     Baseline: {fl['Baseline']['normalized_entropy']:.4f}  |  "
          f"Router: {fl['Router']['normalized_entropy']:.4f}")

    # 7. GSM8K
    gs = all_results["gsm8k"]
    print(f"\n  7. GSM8K End-to-End Accuracy")
    print(f"     Baseline: {gs['Baseline']['accuracy']:.4f} "
          f"({gs['Baseline']['correct']}/{gs['Baseline']['total']})")
    print(f"     Router:   {gs['Router']['accuracy']:.4f} "
          f"({gs['Router']['correct']}/{gs['Router']['total']})")

    # 8. MATH
    mt = all_results["math"]
    print(f"\n  8. MATH End-to-End Accuracy")
    print(f"     Baseline: {mt['Baseline']['accuracy']:.4f} "
          f"({mt['Baseline']['correct']}/{mt['Baseline']['total']})")
    print(f"     Router:   {mt['Router']['accuracy']:.4f} "
          f"({mt['Router']['correct']}/{mt['Router']['total']})")

    print(f"\n{'='*80}")

    # Save all results
    with open("eval_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("Results saved to eval_results.json")
    print("Plots saved: entropy_curve.png, flatness.png, sweep_summary.png")
    print("Stories saved: generated_stories.txt")
