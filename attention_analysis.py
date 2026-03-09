"""
Attention Dilution Analysis
============================
Shows that base LLaDA's self-attention under-attends to local unmasked tokens
at high mask ratios, and that local-unmasked attention correlates with prediction
accuracy — motivating ALA's dedicated local context pathway.

Experiments:
  A. Attention Distribution: fraction of attention going to mask vs unmasked vs
     local-unmasked tokens, swept across mask ratios.
  B. Attention-Accuracy Correlation: bin masked positions by how much attention
     they give to local unmasked tokens, show accuracy increases with attention.

Produces:
  attention_dilution.png         — 2-panel figure for the paper
  attention_by_layer.png         — layer-wise dilution (optional)
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np
import math
import matplotlib.pyplot as plt
import os
from models import ALALLaDA, MASK_TOKEN_ID, ALPHA_BASE, ALPHA_SCALE

RANGE_R = 10  # same as ALA router

# ============================================================
# ATTENTION WEIGHT EXTRACTION
# ============================================================

class AttentionCapture:
    """Monkey-patches _scaled_dot_product_attention on selected blocks
    to compute and store attention weights manually."""

    def __init__(self, blocks, layer_indices):
        self.blocks = blocks
        self.layer_indices = layer_indices
        self.captured = {}  # layer_idx -> [B, nh, T, T] attention weights
        self._originals = {}

    def __enter__(self):
        for idx in self.layer_indices:
            block = self.blocks[idx]
            self._originals[idx] = block._scaled_dot_product_attention
            captured = self.captured
            layer_idx = idx

            def make_hook(li):
                def manual_attn(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
                    # Handle GQA: repeat k,v to match q heads
                    num_kv_heads = k.size(1)
                    num_q_heads = q.size(1)
                    if num_q_heads != num_kv_heads:
                        k_exp = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
                        v_exp = v.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
                    else:
                        k_exp, v_exp = k, v

                    scale = q.size(-1) ** -0.5
                    attn_weights = torch.matmul(q, k_exp.transpose(-2, -1)) * scale
                    if attn_mask is not None:
                        attn_weights = attn_weights + attn_mask
                    attn_weights = F.softmax(attn_weights, dim=-1)
                    captured[li] = attn_weights.detach().cpu().float()
                    return torch.matmul(attn_weights.to(v_exp.dtype), v_exp)
                return manual_attn

            block._scaled_dot_product_attention = make_hook(idx)
        return self

    def __exit__(self, *args):
        for idx in self.layer_indices:
            self.blocks[idx]._scaled_dot_product_attention = self._originals[idx]
        self._originals.clear()


# ============================================================
# SHARED UTILITIES
# ============================================================

def apply_mask(ids, p_mask, device):
    """Apply random mask, never masking position 0 (BOS)."""
    prob = torch.full(ids.shape, p_mask, device=device)
    prob[:, 0] = 0.0
    masked = ids.clone()
    mask_locs = torch.bernoulli(prob).bool()
    masked[mask_locs] = MASK_TOKEN_ID
    return masked, mask_locs


def get_blocks(model):
    """Get transformer blocks from the ALALLaDA model."""
    return model.base_model.model.transformer.blocks


# ============================================================
# EXPERIMENT A: Attention Distribution by Mask Ratio
# ============================================================

@torch.no_grad()
def experiment_attention_distribution(model, tokenizer,
                                       mask_ratios=None,
                                       num_samples=20,
                                       max_length=128,
                                       analyze_layers=None):
    """For each mask ratio, compute how much attention masked positions
    give to: other mask tokens, distant unmasked, local unmasked."""

    if mask_ratios is None:
        mask_ratios = [0.15, 0.30, 0.50, 0.70, 0.85, 0.95]

    blocks = get_blocks(model)
    num_layers = len(blocks)

    if analyze_layers is None:
        # Analyze last 8 layers (where high-level features are)
        analyze_layers = list(range(num_layers - 8, num_layers))

    print("\n" + "=" * 60)
    print("EXPERIMENT A: Attention Distribution by Mask Ratio")
    print(f"  Analyzing layers: {analyze_layers}")
    print("=" * 60)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in dataset["text"][:500] if len(t) > 80][:num_samples]

    results = {p: {"attn_to_mask": [], "attn_to_distant_unmasked": [],
                    "attn_to_local_unmasked": []}
               for p in mask_ratios}

    for p_mask in mask_ratios:
        print(f"  p_mask={p_mask:.2f}...", end=" ", flush=True)

        for text in texts:
            ids = tokenizer(text, return_tensors="pt",
                            truncation=True, max_length=max_length)["input_ids"].to(model.device)
            masked, mask_locs = apply_mask(ids, p_mask, model.device)

            mask_positions = mask_locs[0].nonzero(as_tuple=True)[0].cpu()
            unmasked_positions = (~mask_locs[0]).nonzero(as_tuple=True)[0].cpu()

            if len(mask_positions) < 2 or len(unmasked_positions) < 1:
                continue

            # Capture attention weights from selected layers
            with AttentionCapture(blocks, analyze_layers) as cap:
                model.base_model(masked, output_hidden_states=False)

            # Analyze attention at masked positions, averaged over selected layers and heads
            seq_len = ids.shape[1]
            attn_to_mask_vals = []
            attn_to_distant_vals = []
            attn_to_local_vals = []

            for li in analyze_layers:
                # cap.captured[li] shape: [B, nh, T, T]
                attn = cap.captured[li][0]  # [nh, T, T] — remove batch dim
                # Average across heads
                attn_avg = attn.mean(dim=0)  # [T, T]

                for m_pos in mask_positions:
                    m = m_pos.item()
                    row = attn_avg[m]  # [T] — attention distribution for this masked position

                    # Categorize each position
                    is_mask = mask_locs[0].cpu()
                    is_unmasked = ~is_mask
                    is_local_unmasked = torch.zeros(seq_len, dtype=torch.bool)
                    for u_pos in unmasked_positions:
                        u = u_pos.item()
                        if 0 < abs(u - m) <= RANGE_R:
                            is_local_unmasked[u] = True
                    is_distant_unmasked = is_unmasked & ~is_local_unmasked

                    attn_to_mask_vals.append(row[is_mask].sum().item())
                    attn_to_local_vals.append(row[is_local_unmasked].sum().item())
                    attn_to_distant_vals.append(row[is_distant_unmasked].sum().item())

            results[p_mask]["attn_to_mask"].append(np.mean(attn_to_mask_vals))
            results[p_mask]["attn_to_local_unmasked"].append(np.mean(attn_to_local_vals))
            results[p_mask]["attn_to_distant_unmasked"].append(np.mean(attn_to_distant_vals))

        print("done")

    # Aggregate
    agg = {}
    for p in mask_ratios:
        r = results[p]
        agg[p] = {
            "attn_to_mask": float(np.mean(r["attn_to_mask"])),
            "attn_to_local_unmasked": float(np.mean(r["attn_to_local_unmasked"])),
            "attn_to_distant_unmasked": float(np.mean(r["attn_to_distant_unmasked"])),
        }

    # Print summary
    print(f"\n  {'p_mask':<8} | {'To Mask':>10} | {'To Local Unmask':>16} | {'To Distant Unmask':>18}")
    print(f"  {'-'*58}")
    for p in mask_ratios:
        a = agg[p]
        print(f"  {p:<8.2f} | {a['attn_to_mask']:>10.4f} | "
              f"{a['attn_to_local_unmasked']:>16.4f} | {a['attn_to_distant_unmasked']:>18.4f}")

    return agg


# ============================================================
# EXPERIMENT B: Attention-Accuracy Correlation
# ============================================================

@torch.no_grad()
def experiment_attention_accuracy(model, tokenizer,
                                    mask_ratios=None,
                                    num_samples=30,
                                    max_length=128,
                                    analyze_layers=None,
                                    num_bins=5):
    """For each masked position, measure local-unmasked attention and whether
    the base model predicts correctly. Bin by attention quintile."""

    if mask_ratios is None:
        mask_ratios = [0.50, 0.70, 0.85]

    blocks = get_blocks(model)
    num_layers = len(blocks)

    if analyze_layers is None:
        analyze_layers = list(range(num_layers - 8, num_layers))

    print("\n" + "=" * 60)
    print("EXPERIMENT B: Attention-Accuracy Correlation")
    print(f"  Analyzing layers: {analyze_layers}")
    print("=" * 60)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in dataset["text"][:500] if len(t) > 80][:num_samples]

    # Collect (local_attn, is_correct) pairs across all samples and mask ratios
    all_local_attn = []
    all_correct = []

    for p_mask in mask_ratios:
        print(f"  p_mask={p_mask:.2f}...", end=" ", flush=True)

        for text in texts:
            ids = tokenizer(text, return_tensors="pt",
                            truncation=True, max_length=max_length)["input_ids"].to(model.device)
            original = ids.clone()
            masked, mask_locs = apply_mask(ids, p_mask, model.device)

            mask_positions = mask_locs[0].nonzero(as_tuple=True)[0].cpu()
            unmasked_positions = (~mask_locs[0]).nonzero(as_tuple=True)[0].cpu()

            if len(mask_positions) < 2 or len(unmasked_positions) < 1:
                continue

            # Capture attention + get logits in one forward pass
            with AttentionCapture(blocks, analyze_layers) as cap:
                outputs = model.base_model(masked, output_hidden_states=False)

            logits = outputs.logits  # [B, T, vocab]
            preds = logits.argmax(dim=-1)  # [B, T]

            seq_len = ids.shape[1]

            for m_pos in mask_positions:
                m = m_pos.item()

                # Compute local-unmasked attention (averaged over layers and heads)
                local_attn_vals = []
                for li in analyze_layers:
                    attn_avg = cap.captured[li][0].mean(dim=0)  # [T, T]
                    row = attn_avg[m]

                    local_attn = 0.0
                    for u_pos in unmasked_positions:
                        u = u_pos.item()
                        if 0 < abs(u - m) <= RANGE_R:
                            local_attn += row[u].item()
                    local_attn_vals.append(local_attn)

                avg_local_attn = np.mean(local_attn_vals)
                is_correct = (preds[0, m].item() == original[0, m].item())

                all_local_attn.append(avg_local_attn)
                all_correct.append(int(is_correct))

        print("done")

    all_local_attn = np.array(all_local_attn)
    all_correct = np.array(all_correct)

    # Bin by attention quintile
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(all_local_attn, percentiles)
    bin_accs = []
    bin_labels = []
    bin_counts = []

    print(f"\n  {'Attention Bin':<20} | {'Accuracy':>10} | {'Count':>8}")
    print(f"  {'-'*44}")

    for i in range(num_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == num_bins - 1:
            in_bin = (all_local_attn >= lo) & (all_local_attn <= hi)
        else:
            in_bin = (all_local_attn >= lo) & (all_local_attn < hi)

        if in_bin.sum() == 0:
            bin_accs.append(0)
            bin_counts.append(0)
        else:
            bin_accs.append(all_correct[in_bin].mean())
            bin_counts.append(int(in_bin.sum()))

        label = f"[{lo:.4f}, {hi:.4f})"
        bin_labels.append(f"Q{i+1}")
        print(f"  {label:<20} | {bin_accs[-1]:>10.4f} | {bin_counts[-1]:>8}")

    overall_acc = all_correct.mean()
    print(f"\n  Overall accuracy: {overall_acc:.4f} (n={len(all_correct)})")

    return {
        "bin_labels": bin_labels,
        "bin_accs": bin_accs,
        "bin_counts": bin_counts,
        "all_local_attn": all_local_attn,
        "all_correct": all_correct,
        "overall_acc": overall_acc,
    }


# ============================================================
# EXPERIMENT D: ALA Correction Rate vs Base Attention
# ============================================================

@torch.no_grad()
def experiment_ala_vs_attention(model, tokenizer,
                                 mask_ratios=None,
                                 num_samples=30,
                                 max_length=128,
                                 analyze_layers=None,
                                 num_bins=5):
    """For each masked position, measure base local-unmasked attention AND
    whether ALA flips the prediction from wrong to right.

    If ALA helps more where base attention is most diluted, that's direct
    evidence ALA compensates for attention dilution."""

    if mask_ratios is None:
        mask_ratios = [0.50, 0.70, 0.85]

    blocks = get_blocks(model)
    num_layers = len(blocks)

    if analyze_layers is None:
        analyze_layers = list(range(num_layers - 8, num_layers))

    print("\n" + "=" * 60)
    print("EXPERIMENT D: ALA Correction Rate vs Base Attention")
    print(f"  Analyzing layers: {analyze_layers}")
    print("=" * 60)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in dataset["text"][:500] if len(t) > 80][:num_samples]

    all_local_attn = []
    all_base_correct = []
    all_router_correct = []

    for p_mask in mask_ratios:
        print(f"  p_mask={p_mask:.2f}...", end=" ", flush=True)
        alpha = ALPHA_BASE + ALPHA_SCALE * p_mask

        for text in texts:
            ids = tokenizer(text, return_tensors="pt",
                            truncation=True, max_length=max_length)["input_ids"].to(model.device)
            original = ids.clone()
            masked, mask_locs = apply_mask(ids, p_mask, model.device)

            mask_positions = mask_locs[0].nonzero(as_tuple=True)[0].cpu()
            unmasked_positions = (~mask_locs[0]).nonzero(as_tuple=True)[0].cpu()

            if len(mask_positions) < 2 or len(unmasked_positions) < 1:
                continue

            # Forward pass with attention capture (base model)
            with AttentionCapture(blocks, analyze_layers) as cap:
                base_outputs = model.base_model(masked, output_hidden_states=True)

            base_logits = base_outputs.logits
            base_preds = base_logits.argmax(dim=-1)

            # Router forward pass (uses last hidden state + router)
            h_L = base_outputs.hidden_states[-1].to(torch.bfloat16)
            m_idx = [mask_positions.to(model.device)]
            u_idx = [unmasked_positions.to(model.device)]
            delta = model.router(h_L, m_idx, u_idx)
            h_blended = h_L + alpha * delta

            router_logits = model.base_model.model.transformer.ff_out(h_blended)
            if model.base_model.model.config.scale_logits:
                router_logits *= 1.0 / math.sqrt(model.base_model.model.config.d_model)
            router_preds = router_logits.argmax(dim=-1)

            seq_len = ids.shape[1]

            for m_pos in mask_positions:
                m = m_pos.item()

                # Compute local-unmasked attention
                local_attn_vals = []
                for li in analyze_layers:
                    attn_avg = cap.captured[li][0].mean(dim=0)
                    row = attn_avg[m]
                    local_attn = 0.0
                    for u_pos in unmasked_positions:
                        u = u_pos.item()
                        if 0 < abs(u - m) <= RANGE_R:
                            local_attn += row[u].item()
                    local_attn_vals.append(local_attn)

                avg_local_attn = np.mean(local_attn_vals)
                gold = original[0, m].item()
                base_hit = (base_preds[0, m].item() == gold)
                router_hit = (router_preds[0, m].item() == gold)

                all_local_attn.append(avg_local_attn)
                all_base_correct.append(int(base_hit))
                all_router_correct.append(int(router_hit))

        print("done")

    all_local_attn = np.array(all_local_attn)
    all_base_correct = np.array(all_base_correct)
    all_router_correct = np.array(all_router_correct)

    # ALA correction: base wrong -> router right
    ala_flipped = (~all_base_correct.astype(bool)) & all_router_correct.astype(bool)
    # ALA regression: base right -> router wrong
    ala_regressed = all_base_correct.astype(bool) & (~all_router_correct.astype(bool))

    # Bin by attention quintile
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(all_local_attn, percentiles)
    bin_correction_rate = []
    bin_regression_rate = []
    bin_base_acc = []
    bin_router_acc = []
    bin_labels = []
    bin_counts = []

    print(f"\n  {'Bin':<5} | {'Base Acc':>10} | {'Router Acc':>12} | "
          f"{'Correction':>12} | {'Regression':>12} | {'Count':>8}")
    print(f"  {'-'*68}")

    for i in range(num_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == num_bins - 1:
            in_bin = (all_local_attn >= lo) & (all_local_attn <= hi)
        else:
            in_bin = (all_local_attn >= lo) & (all_local_attn < hi)

        n = int(in_bin.sum())
        bin_counts.append(n)
        bin_labels.append(f"Q{i+1}")

        if n == 0:
            bin_correction_rate.append(0)
            bin_regression_rate.append(0)
            bin_base_acc.append(0)
            bin_router_acc.append(0)
        else:
            bin_correction_rate.append(ala_flipped[in_bin].mean())
            bin_regression_rate.append(ala_regressed[in_bin].mean())
            bin_base_acc.append(all_base_correct[in_bin].mean())
            bin_router_acc.append(all_router_correct[in_bin].mean())

        print(f"  Q{i+1:<4} | {bin_base_acc[-1]:>10.4f} | {bin_router_acc[-1]:>12.4f} | "
              f"{bin_correction_rate[-1]:>12.4f} | {bin_regression_rate[-1]:>12.4f} | {bin_counts[-1]:>8}")

    total_corrections = int(ala_flipped.sum())
    total_regressions = int(ala_regressed.sum())
    print(f"\n  Total corrections (wrong->right): {total_corrections}")
    print(f"  Total regressions (right->wrong): {total_regressions}")
    print(f"  Net improvement: {total_corrections - total_regressions}")

    return {
        "bin_labels": bin_labels,
        "bin_correction_rate": bin_correction_rate,
        "bin_regression_rate": bin_regression_rate,
        "bin_base_acc": bin_base_acc,
        "bin_router_acc": bin_router_acc,
        "bin_counts": bin_counts,
    }


# ============================================================
# EXPERIMENT E: Anchor Availability Analysis
# ============================================================

@torch.no_grad()
def experiment_anchor_availability(model, tokenizer,
                                     mask_ratios=None,
                                     num_samples=30,
                                     max_length=128):
    """For each masked position, count anchors (unmasked tokens within r=10)
    and record whether ALA flips the prediction from wrong to right.

    Bins by anchor count to test: does ALA help more when more local
    context is available? This controls for the ceiling-effect argument
    that Table 2 could be explained by baseline difficulty alone."""

    if mask_ratios is None:
        mask_ratios = [0.50, 0.70, 0.85, 0.95]

    print("\n" + "=" * 60)
    print("EXPERIMENT E: Anchor Availability Analysis")
    print("=" * 60)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in dataset["text"][:500] if len(t) > 80][:num_samples]

    all_anchor_counts = []
    all_base_correct = []
    all_router_correct = []
    all_mask_ratios = []

    for p_mask in mask_ratios:
        print(f"  p_mask={p_mask:.2f}...", end=" ", flush=True)
        alpha = ALPHA_BASE + ALPHA_SCALE * p_mask

        for text in texts:
            ids = tokenizer(text, return_tensors="pt",
                            truncation=True, max_length=max_length)["input_ids"].to(model.device)
            original = ids.clone()
            masked, mask_locs = apply_mask(ids, p_mask, model.device)

            mask_positions = mask_locs[0].nonzero(as_tuple=True)[0].cpu()
            unmasked_positions = (~mask_locs[0]).nonzero(as_tuple=True)[0].cpu()

            if len(mask_positions) < 2 or len(unmasked_positions) < 1:
                continue

            # Base forward pass
            base_outputs = model.base_model(masked, output_hidden_states=True)
            base_logits = base_outputs.logits
            base_preds = base_logits.argmax(dim=-1)

            # Router forward pass
            h_L = base_outputs.hidden_states[-1].to(torch.bfloat16)
            m_idx = [mask_positions.to(model.device)]
            u_idx = [unmasked_positions.to(model.device)]
            delta = model.router(h_L, m_idx, u_idx)
            h_blended = h_L + alpha * delta

            router_logits = model.base_model.model.transformer.ff_out(h_blended)
            if model.base_model.model.config.scale_logits:
                router_logits *= 1.0 / math.sqrt(model.base_model.model.config.d_model)
            router_preds = router_logits.argmax(dim=-1)

            # For each masked position, count anchors within r
            unmasked_set = set(unmasked_positions.tolist())
            for m_pos in mask_positions:
                m = m_pos.item()
                anchor_count = sum(
                    1 for u in unmasked_set if 0 < abs(u - m) <= RANGE_R
                )
                gold = original[0, m].item()
                base_hit = (base_preds[0, m].item() == gold)
                router_hit = (router_preds[0, m].item() == gold)

                all_anchor_counts.append(anchor_count)
                all_base_correct.append(int(base_hit))
                all_router_correct.append(int(router_hit))
                all_mask_ratios.append(p_mask)

        print("done")

    all_anchor_counts = np.array(all_anchor_counts)
    all_base_correct = np.array(all_base_correct)
    all_router_correct = np.array(all_router_correct)
    all_mask_ratios = np.array(all_mask_ratios)

    ala_flipped = (~all_base_correct.astype(bool)) & all_router_correct.astype(bool)
    ala_regressed = all_base_correct.astype(bool) & (~all_router_correct.astype(bool))

    # --- Summary by anchor count ---
    unique_counts = sorted(np.unique(all_anchor_counts))
    print(f"\n  {'Anchors':<10} | {'Base Acc':>10} | {'Router Acc':>12} | "
          f"{'Correction':>12} | {'Regression':>12} | {'Delta Acc':>10} | {'Count':>8}")
    print(f"  {'-'*82}")

    bin_anchor_counts = []
    bin_base_acc = []
    bin_router_acc = []
    bin_delta_acc = []
    bin_correction_rate = []
    bin_regression_rate = []
    bin_counts = []

    for ac in unique_counts:
        in_bin = all_anchor_counts == ac
        n = int(in_bin.sum())
        if n < 10:  # skip tiny bins
            continue

        b_acc = all_base_correct[in_bin].mean()
        r_acc = all_router_correct[in_bin].mean()
        corr = ala_flipped[in_bin].mean()
        reg = ala_regressed[in_bin].mean()

        bin_anchor_counts.append(ac)
        bin_base_acc.append(b_acc)
        bin_router_acc.append(r_acc)
        bin_delta_acc.append(r_acc - b_acc)
        bin_correction_rate.append(corr)
        bin_regression_rate.append(reg)
        bin_counts.append(n)

        print(f"  {ac:<10} | {b_acc:>10.4f} | {r_acc:>12.4f} | "
              f"{corr:>12.4f} | {reg:>12.4f} | {r_acc - b_acc:>+10.4f} | {n:>8}")

    # --- Summary by mask ratio (with avg anchor count) ---
    print(f"\n  {'p_mask':<8} | {'Avg Anchors':>12} | {'Base Acc':>10} | "
          f"{'Router Acc':>12} | {'Delta':>8}")
    print(f"  {'-'*56}")
    for p in mask_ratios:
        in_p = all_mask_ratios == p
        avg_anc = all_anchor_counts[in_p].mean()
        b_acc = all_base_correct[in_p].mean()
        r_acc = all_router_correct[in_p].mean()
        print(f"  {p:<8.2f} | {avg_anc:>12.1f} | {b_acc:>10.4f} | "
              f"{r_acc:>12.4f} | {r_acc - b_acc:>+8.4f}")

    return {
        "bin_anchor_counts": bin_anchor_counts,
        "bin_base_acc": bin_base_acc,
        "bin_router_acc": bin_router_acc,
        "bin_delta_acc": bin_delta_acc,
        "bin_correction_rate": bin_correction_rate,
        "bin_regression_rate": bin_regression_rate,
        "bin_counts": bin_counts,
        "all_anchor_counts": all_anchor_counts,
        "all_base_correct": all_base_correct,
        "all_router_correct": all_router_correct,
        "all_mask_ratios": all_mask_ratios,
    }


def plot_anchor_availability(anchor_results, save_path="anchor_availability.png"):
    """2-panel figure: (a) correction rate vs anchor count, (b) delta accuracy vs anchor count."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    counts = anchor_results["bin_anchor_counts"]
    correction = anchor_results["bin_correction_rate"]
    regression = anchor_results["bin_regression_rate"]
    delta_acc = anchor_results["bin_delta_acc"]
    base_acc = anchor_results["bin_base_acc"]
    n_bins = anchor_results["bin_counts"]

    # --- Panel (a): Correction rate vs anchor count ---
    ax1 = axes[0]
    x = np.arange(len(counts))
    width = 0.35
    ax1.bar(x - width/2, correction, width, label='Corrections (wrong→right)',
            color='#2ca02c', alpha=0.8)
    ax1.bar(x + width/2, regression, width, label='Regressions (right→wrong)',
            color='#d62728', alpha=0.8)
    ax1.set_xlabel('Number of Anchors (unmasked tokens within r=10)', fontsize=11)
    ax1.set_ylabel('Rate', fontsize=11)
    ax1.set_title('(a) ALA Correction Rate vs. Anchor Count', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(counts)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # Annotate counts
    for i, (bar_x, n) in enumerate(zip(x, n_bins)):
        ax1.text(bar_x, max(correction[i], regression[i]) + 0.005,
                 f'n={n}', ha='center', va='bottom', fontsize=7, color='gray')

    # --- Panel (b): Delta accuracy vs anchor count ---
    ax2 = axes[1]
    colors = ['#d62728' if d < 0 else '#2ca02c' for d in delta_acc]
    bars = ax2.bar(x, delta_acc, 0.6, color=colors, alpha=0.8)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_xlabel('Number of Anchors (unmasked tokens within r=10)', fontsize=11)
    ax2.set_ylabel('Accuracy Improvement (Router − Base)', fontsize=11)
    ax2.set_title('(b) ALA Accuracy Gain vs. Anchor Count', fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(counts)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add base accuracy as secondary info
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x, base_acc, 'ko--', markersize=4, alpha=0.5, label='Base accuracy')
    ax2_twin.set_ylabel('Base Accuracy', fontsize=10, color='gray')
    ax2_twin.tick_params(axis='y', labelcolor='gray')
    ax2_twin.legend(fontsize=8, loc='upper right')

    for bar, d in zip(bars, delta_acc):
        y = d + (0.002 if d >= 0 else -0.006)
        ax2.text(bar.get_x() + bar.get_width()/2, y,
                 f'{d:+.1%}', ha='center', va='bottom' if d >= 0 else 'top',
                 fontsize=8, fontweight='bold')

    plt.suptitle(
        'Anchor availability analysis: ALA extracts more value when more local context is available',
        fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {save_path}")


# ============================================================
# EXPERIMENT C: Layer-wise Attention Dilution
# ============================================================

@torch.no_grad()
def experiment_layerwise_dilution(model, tokenizer,
                                    p_mask=0.85,
                                    num_samples=10,
                                    max_length=128):
    """At a fixed mask ratio, compute local-unmasked attention per layer."""

    blocks = get_blocks(model)
    num_layers = len(blocks)
    all_layers = list(range(num_layers))

    print("\n" + "=" * 60)
    print(f"EXPERIMENT C: Layer-wise Attention Dilution (p_mask={p_mask})")
    print("=" * 60)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in dataset["text"][:500] if len(t) > 80][:num_samples]

    # Process in chunks of layers to manage memory
    chunk_size = 8
    layer_local_attn = np.zeros(num_layers)
    layer_mask_attn = np.zeros(num_layers)
    n_valid = 0

    for text in texts:
        ids = tokenizer(text, return_tensors="pt",
                        truncation=True, max_length=max_length)["input_ids"].to(model.device)
        masked, mask_locs = apply_mask(ids, p_mask, model.device)

        mask_positions = mask_locs[0].nonzero(as_tuple=True)[0].cpu()
        unmasked_positions = (~mask_locs[0]).nonzero(as_tuple=True)[0].cpu()

        if len(mask_positions) < 2 or len(unmasked_positions) < 1:
            continue

        seq_len = ids.shape[1]

        # Precompute local-unmasked boolean per masked position
        local_masks = {}
        for m_pos in mask_positions:
            m = m_pos.item()
            local = torch.zeros(seq_len, dtype=torch.bool)
            for u_pos in unmasked_positions:
                u = u_pos.item()
                if 0 < abs(u - m) <= RANGE_R:
                    local[u] = True
            local_masks[m] = local

        is_mask_bool = mask_locs[0].cpu()

        # Process layers in chunks
        for chunk_start in range(0, num_layers, chunk_size):
            chunk_layers = list(range(chunk_start, min(chunk_start + chunk_size, num_layers)))

            with AttentionCapture(blocks, chunk_layers) as cap:
                model.base_model(masked, output_hidden_states=False)

            for li in chunk_layers:
                attn_avg = cap.captured[li][0].mean(dim=0)  # [T, T]

                local_vals = []
                mask_vals = []
                for m_pos in mask_positions:
                    m = m_pos.item()
                    row = attn_avg[m]
                    local_vals.append(row[local_masks[m]].sum().item())
                    mask_vals.append(row[is_mask_bool].sum().item())

                layer_local_attn[li] += np.mean(local_vals)
                layer_mask_attn[li] += np.mean(mask_vals)

        n_valid += 1
        print(f"  Sample {n_valid}/{num_samples}")

    if n_valid > 0:
        layer_local_attn /= n_valid
        layer_mask_attn /= n_valid

    print(f"\n  Layer 0:  local={layer_local_attn[0]:.4f}, mask={layer_mask_attn[0]:.4f}")
    print(f"  Layer {num_layers//2}: local={layer_local_attn[num_layers//2]:.4f}, mask={layer_mask_attn[num_layers//2]:.4f}")
    print(f"  Layer {num_layers-1}: local={layer_local_attn[-1]:.4f}, mask={layer_mask_attn[-1]:.4f}")

    return {
        "layer_local_attn": layer_local_attn,
        "layer_mask_attn": layer_mask_attn,
        "num_layers": num_layers,
    }


# ============================================================
# PLOTTING
# ============================================================

def plot_attention_dilution(attn_dist, attn_acc, ala_vs_attn,
                            save_path="attention_dilution.png"):
    """3-panel figure for the paper."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Panel 1: Attention budget breakdown ---
    ax1 = axes[0]
    mask_ratios = sorted(attn_dist.keys())
    to_mask = [attn_dist[p]["attn_to_mask"] for p in mask_ratios]
    to_local = [attn_dist[p]["attn_to_local_unmasked"] for p in mask_ratios]
    to_distant = [attn_dist[p]["attn_to_distant_unmasked"] for p in mask_ratios]

    x = np.arange(len(mask_ratios))
    width = 0.6
    ax1.bar(x, to_mask, width, label='To [MASK] tokens', color='#d62728', alpha=0.8)
    ax1.bar(x, to_distant, width, bottom=to_mask, label='To distant unmasked', color='#1f77b4', alpha=0.8)
    bottom2 = [m + d for m, d in zip(to_mask, to_distant)]
    ax1.bar(x, to_local, width, bottom=bottom2, label='To local unmasked (r$\\leq$10)', color='#2ca02c', alpha=0.8)

    ax1.set_xlabel('Mask Ratio ($p_{mask}$)', fontsize=12)
    ax1.set_ylabel('Fraction of Attention', fontsize=12)
    ax1.set_title('(a) Attention Budget at Masked Positions', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{p:.0%}' for p in mask_ratios])
    ax1.legend(fontsize=8, loc='upper left')
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3, axis='y')

    # Annotate local attention at high mask ratio
    high_p_idx = -1
    ax1.annotate(
        f'Only {to_local[high_p_idx]:.1%} to\nlocal unmasked',
        xy=(x[high_p_idx], bottom2[high_p_idx] + to_local[high_p_idx] / 2),
        xytext=(x[high_p_idx] - 1.5, 0.5),
        fontsize=9, color='#2ca02c', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.5))

    # --- Panel 2: Accuracy vs local attention quintile ---
    ax2 = axes[1]
    bin_labels = attn_acc["bin_labels"]
    bin_accs = attn_acc["bin_accs"]
    bin_counts = attn_acc["bin_counts"]

    colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(bin_labels)))
    bars = ax2.bar(bin_labels, bin_accs, color=colors, edgecolor='white', linewidth=1.5)

    ax2.set_xlabel('Local-Unmasked Attention Quintile', fontsize=12)
    ax2.set_ylabel('Prediction Accuracy', fontsize=12)
    ax2.set_title('(b) Base Accuracy vs. Local Attention', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, acc, cnt in zip(bars, bin_accs, bin_counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, acc + 0.01,
                 f'{acc:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.axhline(y=attn_acc["overall_acc"], color='gray', linestyle='--',
                alpha=0.6, label=f'Overall: {attn_acc["overall_acc"]:.1%}')
    ax2.legend(fontsize=9)

    # --- Panel 3: ALA correction rate vs local attention ---
    ax3 = axes[2]
    d_labels = ala_vs_attn["bin_labels"]
    correction = ala_vs_attn["bin_correction_rate"]
    regression = ala_vs_attn["bin_regression_rate"]

    x3 = np.arange(len(d_labels))
    width3 = 0.35
    bars_corr = ax3.bar(x3 - width3/2, correction, width3,
                         label='Corrections (wrong$\\to$right)', color='#2ca02c', alpha=0.8)
    bars_reg = ax3.bar(x3 + width3/2, regression, width3,
                        label='Regressions (right$\\to$wrong)', color='#d62728', alpha=0.8)

    ax3.set_xlabel('Local-Unmasked Attention Quintile', fontsize=12)
    ax3.set_ylabel('Rate', fontsize=12)
    ax3.set_title('(c) ALA Correction Rate vs. Base Attention', fontsize=11)
    ax3.set_xticks(x3)
    ax3.set_xticklabels(d_labels)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    for bar, rate in zip(bars_corr, correction):
        if rate > 0:
            ax3.text(bar.get_x() + bar.get_width() / 2, rate + 0.003,
                     f'{rate:.1%}', ha='center', va='bottom', fontsize=8)

    plt.suptitle(
        'Attention dilution motivates ALA: base model under-attends to local context, '
        'ALA compensates where attention is weakest',
        fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {save_path}")


def plot_layerwise(layer_results, save_path="attention_by_layer.png"):
    """Layer-wise attention dilution plot."""
    fig, ax = plt.subplots(figsize=(14, 5))

    layers = np.arange(layer_results["num_layers"])
    local_attn = layer_results["layer_local_attn"]
    mask_attn = layer_results["layer_mask_attn"]

    ax.plot(layers, mask_attn, color='#d62728', linewidth=2, label='Attention to [MASK] tokens')
    ax.fill_between(layers, mask_attn, alpha=0.15, color='#d62728')
    ax.plot(layers, local_attn, color='#2ca02c', linewidth=2, label='Attention to local unmasked (r≤10)')
    ax.fill_between(layers, local_attn, alpha=0.15, color='#2ca02c')

    ax.set_xlabel('Transformer Layer', fontsize=12)
    ax.set_ylabel('Fraction of Attention', fontsize=12)
    ax.set_title('Attention Allocation Across Layers (p_mask=0.85)\n'
                 'Masked positions attend mostly to other [MASK] tokens, not local context',
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    model_id = 'GSAI-ML/LLaDA-8B-Instruct'
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

    # Experiment A: Attention distribution by mask ratio
    print("\n" + "=" * 70)
    print("Running Experiment A: Attention Distribution")
    print("=" * 70)
    attn_dist = experiment_attention_distribution(
        model, tokenizer, num_samples=20, max_length=128
    )

    # Experiment B: Attention-accuracy correlation
    print("\n" + "=" * 70)
    print("Running Experiment B: Attention-Accuracy Correlation")
    print("=" * 70)
    attn_acc = experiment_attention_accuracy(
        model, tokenizer, num_samples=30, max_length=128
    )

    # Experiment D: ALA correction rate vs base attention
    print("\n" + "=" * 70)
    print("Running Experiment D: ALA Correction Rate vs Base Attention")
    print("=" * 70)
    ala_vs_attn = experiment_ala_vs_attention(
        model, tokenizer, num_samples=30, max_length=128
    )

    # Plot main figure
    plot_attention_dilution(attn_dist, attn_acc, ala_vs_attn)

    # Experiment E: Anchor availability
    print("\n" + "=" * 70)
    print("Running Experiment E: Anchor Availability Analysis")
    print("=" * 70)
    anchor_results = experiment_anchor_availability(
        model, tokenizer, num_samples=30, max_length=128
    )
    plot_anchor_availability(anchor_results)

    # Experiment C: Layer-wise (optional)
    print("\n" + "=" * 70)
    print("Running Experiment C: Layer-wise Dilution")
    print("=" * 70)
    layer_results = experiment_layerwise_dilution(
        model, tokenizer, p_mask=0.85, num_samples=10
    )
    plot_layerwise(layer_results)

    print("\n" + "=" * 70)
    print("DONE. Plots saved:")
    print("  attention_dilution.png    — main 3-panel figure for paper")
    print("  anchor_availability.png   — anchor count analysis")
    print("  attention_by_layer.png    — layer-wise dilution")
    print("=" * 70)
