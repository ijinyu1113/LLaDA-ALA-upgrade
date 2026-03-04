"""
Cosine Similarity Collapse Experiment
======================================
Measures pairwise cosine similarity between hidden states at masked positions,
comparing:
  - Base LLaDA hidden states (h_L from frozen model)
  - ALA Router-corrected hidden states (h_blended = (1-α)*h_L + α*Δh)

The hypothesis: at high mask ratios, base model masked-position hidden states
collapse (cosine sim → 1.0), while ALA-corrected states remain differentiated.

Produces two plots:
  1. cosine_sim_by_mask_ratio.png  — mean cosine sim vs p_mask for base vs router
  2. cosine_sim_by_layer.png       — mean cosine sim across transformer layers
                                     at a fixed high mask ratio (p=0.85)
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import os
from models import AMIPRouterInference, ALALLaDA, ALPHA_BASE, ALPHA_SCALE, MASK_TOKEN_ID


# ============================================================
# CORE MEASUREMENT FUNCTIONS
# ============================================================

def mean_pairwise_cosine_sim(hidden_states):
    """
    Compute mean pairwise cosine similarity over a set of vectors.
    hidden_states: [N, d_model]
    Returns: scalar float
    """
    if hidden_states.shape[0] < 2:
        return float('nan')
    h = F.normalize(hidden_states.float(), dim=-1)     # [N, d_model]
    sim_matrix = h @ h.T                               # [N, N]
    # Only upper triangle (exclude diagonal)
    N = h.shape[0]
    mask = torch.triu(torch.ones(N, N, dtype=torch.bool, device=h.device), diagonal=1)
    return sim_matrix[mask].mean().item()


@torch.no_grad()
def get_base_and_router_hidden_states(model, input_ids, alpha=0.1):
    """
    Returns:
      h_base:   final layer hidden states at masked positions  [N_masked, d_model]
      h_router: router-corrected hidden states at masked positions [N_masked, d_model]
      all_layer_hidden_states: list of [N_masked, d_model] per layer (base only)
    """
    mask_token_id = MASK_TOKEN_ID
    outputs = model.base_model(input_ids, output_hidden_states=True)

    # All layers: outputs.hidden_states is a tuple of (n_layers+1) tensors
    all_hidden = outputs.hidden_states   # each: [1, seq_len, d_model]

    mask_pos = (input_ids[0] == mask_token_id).nonzero(as_tuple=True)[0]
    if len(mask_pos) == 0:
        return None, None, None

    # Base: final layer at masked positions
    h_base = all_hidden[-1][0, mask_pos, :].to(torch.float32)  # [N_masked, d_model]

    # Router correction
    h_L = all_hidden[-1].to(torch.bfloat16)
    m_idx_list = [mask_pos]
    u_idx_list = [(input_ids[0] != mask_token_id).nonzero(as_tuple=True)[0]]
    delta = model.router(h_L, m_idx_list, u_idx_list)
    h_blended = (1 - alpha) * h_L + alpha * delta
    h_router = h_blended[0, mask_pos, :].to(torch.float32)  # [N_masked, d_model]

    # Per-layer base hidden states at masked positions
    layer_states = [layer[0, mask_pos, :].to(torch.float32) for layer in all_hidden]

    return h_base, h_router, layer_states


# ============================================================
# EXPERIMENT 1: Cosine Sim vs Mask Ratio
# ============================================================
@torch.no_grad()
def experiment_cosine_by_mask_ratio(model, tokenizer,
                                     mask_ratios=None,
                                     num_samples=20,
                                     max_length=128):
    """
    For each mask ratio, compute mean pairwise cosine similarity of masked
    position hidden states — base vs router-corrected.
    """
    if mask_ratios is None:
        mask_ratios = [0.15, 0.30, 0.50, 0.70, 0.85, 0.95]

    print("\n" + "="*60)
    print("EXPERIMENT 1: Cosine Similarity vs Mask Ratio")
    print("="*60)
    print(f"  {'p_mask':<8} | {'Base CosSim':<14} | {'Router CosSim':<14} | {'Δ':<10}")
    print(f"  {'-'*52}")

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in dataset["text"][:500] if len(t) > 80][:num_samples]

    base_sims, router_sims = [], []
    for p_mask in mask_ratios:
        batch_base, batch_router = [], []
        alpha = ALPHA_BASE + ALPHA_SCALE * p_mask

        for text in texts:
            enc = tokenizer(text, return_tensors="pt",
                            truncation=True, max_length=max_length)
            ids = enc["input_ids"].to(model.device)

            # Apply mask
            prob = torch.full(ids.shape, p_mask, device=ids.device)
            prob[:, 0] = 0.0
            masked = ids.clone()
            mask_locs = torch.bernoulli(prob).bool()
            masked[mask_locs] = MASK_TOKEN_ID

            if mask_locs.sum() < 2:
                continue

            h_base, h_router, _ = get_base_and_router_hidden_states(model, masked, alpha=alpha)
            if h_base is None:
                continue

            batch_base.append(mean_pairwise_cosine_sim(h_base))
            batch_router.append(mean_pairwise_cosine_sim(h_router))

        mean_base = float(np.nanmean(batch_base)) if batch_base else float('nan')
        mean_router = float(np.nanmean(batch_router)) if batch_router else float('nan')
        base_sims.append(mean_base)
        router_sims.append(mean_router)

        print(f"  {p_mask:<8.2f} | {mean_base:<14.4f} | {mean_router:<14.4f} | "
              f"{mean_router - mean_base:<+10.4f}")

    return mask_ratios, base_sims, router_sims


# ============================================================
# EXPERIMENT 2: Cosine Sim Across Transformer Layers
# ============================================================
@torch.no_grad()
def experiment_cosine_by_layer(model, tokenizer,
                                 p_mask=0.85,
                                 num_samples=10,
                                 max_length=128):
    """
    At a fixed high mask ratio, track how cosine similarity evolves
    layer-by-layer through the transformer (base model only).

    Shows: early layers = nearly identical hidden states (collapse),
           later layers = some differentiation (but still high).
    ALA's final-layer correction shown as a single point.
    """
    print("\n" + "="*60)
    print(f"EXPERIMENT 2: Cosine Similarity by Layer (p_mask={p_mask})")
    print("="*60)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in dataset["text"][:500] if len(t) > 80][:num_samples]

    all_layer_sims = None
    router_final_sims = []
    n_valid = 0

    for text in texts:
        enc = tokenizer(text, return_tensors="pt",
                        truncation=True, max_length=max_length)
        ids = enc["input_ids"].to(model.device)

        prob = torch.full(ids.shape, p_mask, device=ids.device)
        prob[:, 0] = 0.0
        masked = ids.clone()
        mask_locs = torch.bernoulli(prob).bool()
        masked[mask_locs] = MASK_TOKEN_ID

        if mask_locs.sum() < 2:
            continue

        alpha = ALPHA_BASE + ALPHA_SCALE * p_mask
        h_base, h_router, layer_states = get_base_and_router_hidden_states(model, masked, alpha=alpha)
        if h_base is None or layer_states is None:
            continue

        layer_sims = [mean_pairwise_cosine_sim(ls) for ls in layer_states]
        router_final_sims.append(mean_pairwise_cosine_sim(h_router))

        if all_layer_sims is None:
            all_layer_sims = np.array(layer_sims)
        else:
            all_layer_sims += np.array(layer_sims)
        n_valid += 1

    if n_valid == 0:
        print("  No valid samples.")
        return None, None

    avg_layer_sims = all_layer_sims / n_valid
    avg_router_final = float(np.mean(router_final_sims))

    print(f"  Layer 0  (embedding): {avg_layer_sims[0]:.4f}")
    print(f"  Layer 16 (mid):       {avg_layer_sims[len(avg_layer_sims)//2]:.4f}")
    print(f"  Layer -1 (final):     {avg_layer_sims[-1]:.4f}")
    print(f"  Router corrected:     {avg_router_final:.4f}  ← after ALA")

    return avg_layer_sims, avg_router_final


# ============================================================
# PLOTTING
# ============================================================

def plot_cosine_by_mask_ratio(mask_ratios, base_sims, router_sims,
                               save_path="cosine_sim_by_mask_ratio.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: absolute cosine similarity
    ax = axes[0]
    ax.plot(mask_ratios, base_sims, 'o-', color='steelblue', linewidth=2,
            markersize=8, label='Base LLaDA')
    ax.plot(mask_ratios, router_sims, 's-', color='coral', linewidth=2,
            markersize=8, label='ALA Router')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4, label='Perfect collapse (1.0)')
    ax.set_xlabel('Mask Ratio (p_mask)', fontsize=12)
    ax.set_ylabel('Mean Pairwise Cosine Similarity', fontsize=12)
    ax.set_title('Hidden State Collapse at Masked Positions\n(higher = more collapsed)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.05)
    ax.set_xticks(mask_ratios)

    # Right: delta (how much ALA differentiates)
    ax2 = axes[1]
    deltas = [r - b for r, b in zip(router_sims, base_sims)]
    colors = ['coral' if d < 0 else 'steelblue' for d in deltas]  # negative = more differentiated
    bars = ax2.bar([str(p) for p in mask_ratios], deltas, color=colors, alpha=0.8, edgecolor='white')
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.set_xlabel('Mask Ratio (p_mask)', fontsize=12)
    ax2.set_ylabel('Δ Cosine Similarity (Router − Base)', fontsize=12)
    ax2.set_title('ALA Differentiation Effect\n(negative = less collapse = better)', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')

    # Annotate bars
    for bar, d in zip(bars, deltas):
        ax2.text(bar.get_x() + bar.get_width()/2, d - 0.001 if d < 0 else d + 0.001,
                 f'{d:+.4f}', ha='center', va='top' if d < 0 else 'bottom',
                 fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved: {save_path}")


def plot_cosine_by_layer(avg_layer_sims, avg_router_final,
                          p_mask=0.85,
                          save_path="cosine_sim_by_layer.png"):
    if avg_layer_sims is None:
        return

    fig, ax = plt.subplots(figsize=(14, 5))

    layers = np.arange(len(avg_layer_sims))
    ax.plot(layers, avg_layer_sims, color='steelblue', linewidth=2,
            label='Base LLaDA (layer-by-layer)')
    ax.fill_between(layers, avg_layer_sims, alpha=0.15, color='steelblue')

    # Mark the final layer point
    ax.scatter([layers[-1]], [avg_layer_sims[-1]], color='steelblue',
               s=100, zorder=5)

    # Mark ALA-corrected as a star at the final layer
    ax.scatter([layers[-1] + 0.5], [avg_router_final], color='coral',
               s=200, marker='*', zorder=6,
               label=f'ALA Router (after correction): {avg_router_final:.4f}')

    ax.annotate(f'Base final: {avg_layer_sims[-1]:.4f}',
                xy=(layers[-1], avg_layer_sims[-1]),
                xytext=(-40, 15), textcoords='offset points',
                fontsize=9, color='steelblue',
                arrowprops=dict(arrowstyle='->', color='steelblue'))

    ax.annotate(f'ALA corrected: {avg_router_final:.4f}',
                xy=(layers[-1] + 0.5, avg_router_final),
                xytext=(10, -20), textcoords='offset points',
                fontsize=9, color='coral',
                arrowprops=dict(arrowstyle='->', color='coral'))

    ax.set_xlabel('Transformer Layer', fontsize=12)
    ax.set_ylabel('Mean Pairwise Cosine Similarity', fontsize=12)
    ax.set_title(
        f'Symmetry Breaking Across Layers (p_mask={p_mask})\n'
        f'Layer 0 = embedding output; Layer {len(layers)-1} = final hidden state',
        fontsize=12
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.05)

    # Shade "still collapsed" region
    ax.axhspan(0.95, 1.05, alpha=0.07, color='red',
               label='Collapse zone (sim > 0.95)')
    ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.4, linewidth=1)
    ax.text(1, 0.955, 'Collapse zone (sim > 0.95)', color='red',
            fontsize=8, alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Plot saved: {save_path}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # ── Load model (same as eval.py) ──────────────────────────
    model_id = 'GSAI-ML/LLaDA-8B-Instruct'
    print("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model = ALALLaDA(base_model).to(torch.bfloat16)
    device = next(base_model.parameters()).device
    model.router.to(device)

    # ── Load trained router weights ───────────────────────────
    weights_path = "amip_router_final.pt"
    if os.path.exists(weights_path):
        model.router.load_state_dict(
            torch.load(weights_path, map_location=device)
        )
        print(f"Router loaded from {weights_path}")
    else:
        print("WARNING: No saved router found — using random weights.")

    model.eval()

    # ── Run experiments ───────────────────────────────────────
    # Experiment 1: Cosine sim vs mask ratio
    mask_ratios, base_sims, router_sims = experiment_cosine_by_mask_ratio(
        model, tokenizer,
        mask_ratios=[0.15, 0.30, 0.50, 0.70, 0.85, 0.95],
        num_samples=20
    )

    # Experiment 2: Layer-by-layer at p=0.85
    avg_layer_sims, avg_router_final = experiment_cosine_by_layer(
        model, tokenizer,
        p_mask=0.85,
        num_samples=10
    )

    # ── Plots ─────────────────────────────────────────────────
    plot_cosine_by_mask_ratio(mask_ratios, base_sims, router_sims)
    plot_cosine_by_layer(avg_layer_sims, avg_router_final, p_mask=0.85)

    # ── Print summary ─────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY: Cosine Similarity Collapse")
    print("="*60)
    print(f"\n  Mask Ratio | Base CosSim | Router CosSim | Δ")
    print(f"  {'-'*52}")
    for p, b, r in zip(mask_ratios, base_sims, router_sims):
        print(f"  {p:<10.2f} | {b:<11.4f} | {r:<13.4f} | {r-b:<+.4f}")

    print("\n  Interpretation:")
    print("  • Values close to 1.0 = hidden states are nearly identical (collapsed)")
    print("  • Router values below base = ALA is successfully differentiating positions")
    print("  • Layer plot shows WHERE in the transformer collapse occurs")
    print("\nDone. Plots saved.")
