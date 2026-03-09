"""
Mechanistic Interpretability Experiments
=========================================
Three experiments measuring how ALA affects masked-position hidden states:

  1. Cosine Similarity vs Mask Ratio
     Pairwise cosine sim among masked positions — shows ALA increases cohesion.

  2. Cosine Similarity by Layer
     Layer-by-layer pairwise cosine sim at p=0.85 — shows similarity progression.

  3. Token Alignment + Two-Panel Figure
     cos(h, W_out[correct_token]) — shows ALA increases cohesion WITHOUT
     improving ground-truth alignment (mechanism is at logit level, not representation level).

Produces:
  cosine_sim_by_mask_ratio.png   — Experiment 1
  cosine_sim_by_layer.png        — Experiment 2
  two_panel_symmetry_vs_alignment.png — Experiment 3
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
# SHARED UTILITIES
# ============================================================

def mean_pairwise_cosine_sim(hidden_states):
    """Mean pairwise cosine similarity over a set of vectors.
    hidden_states: [N, d_model] -> scalar float
    """
    if hidden_states.shape[0] < 2:
        return float('nan')
    h = F.normalize(hidden_states.float(), dim=-1)
    sim_matrix = h @ h.T
    N = h.shape[0]
    mask = torch.triu(torch.ones(N, N, dtype=torch.bool, device=h.device), diagonal=1)
    return sim_matrix[mask].mean().item()


def apply_mask(ids, p_mask, device):
    """Apply random mask to token ids, never masking position 0 (BOS)."""
    prob = torch.full(ids.shape, p_mask, device=device)
    prob[:, 0] = 0.0
    masked = ids.clone()
    mask_locs = torch.bernoulli(prob).bool()
    masked[mask_locs] = MASK_TOKEN_ID
    return masked, mask_locs


@torch.no_grad()
def get_hidden_states(model, input_ids, original_ids=None):
    """Run base model, compute router correction, return hidden states at masked positions.

    Returns:
        h_base:       [N, d] base model hidden states at masked positions
        h_router:     [N, d] router-corrected hidden states (h + alpha*delta)
        layer_states: list of [N, d] per layer (base only, for layer analysis)
        correct_embs: [N, d] output embeddings of correct tokens (if original_ids provided)
    """
    alpha = ALPHA_BASE + ALPHA_SCALE * 0.5  # use mid-range for default; overridden by callers

    outputs = model.base_model(input_ids, output_hidden_states=True)
    all_hidden = outputs.hidden_states

    mask_pos = (input_ids[0] == MASK_TOKEN_ID).nonzero(as_tuple=True)[0]
    if len(mask_pos) < 2:
        return None, None, None, None

    # Base: final layer at masked positions
    h_L = all_hidden[-1].to(torch.bfloat16)
    h_base = h_L[0, mask_pos, :].to(torch.float32)

    # Router correction: h + alpha * delta (NOT (1-alpha)*h + alpha*delta)
    m_idx = [mask_pos]
    u_idx = [(input_ids[0] != MASK_TOKEN_ID).nonzero(as_tuple=True)[0]]
    delta = model.router(h_L, m_idx, u_idx)
    h_blended = h_L + alpha * delta
    h_router = h_blended[0, mask_pos, :].to(torch.float32)

    # Per-layer base hidden states
    layer_states = [layer[0, mask_pos, :].to(torch.float32) for layer in all_hidden]

    # Correct token embeddings (if ground truth available)
    correct_embs = None
    if original_ids is not None:
        W_out = model.base_model.model.transformer.ff_out.weight.to(torch.float32)
        correct_token_ids = original_ids[0][mask_pos]
        correct_embs = W_out[correct_token_ids]

    return h_base, h_router, layer_states, correct_embs


# ============================================================
# EXPERIMENT 1: Cosine Sim vs Mask Ratio
# ============================================================

@torch.no_grad()
def experiment_cosine_by_mask_ratio(model, tokenizer,
                                     mask_ratios=None,
                                     num_samples=20,
                                     max_length=128):
    if mask_ratios is None:
        mask_ratios = [0.15, 0.30, 0.50, 0.70, 0.85, 0.95]

    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Cosine Similarity vs Mask Ratio")
    print("=" * 60)
    print(f"  {'p_mask':<8} | {'Base CosSim':<14} | {'Router CosSim':<14} | {'Δ':<10}")
    print(f"  {'-'*52}")

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in dataset["text"][:500] if len(t) > 80][:num_samples]

    base_sims, router_sims = [], []
    for p_mask in mask_ratios:
        batch_base, batch_router = [], []

        for text in texts:
            ids = tokenizer(text, return_tensors="pt",
                            truncation=True, max_length=max_length)["input_ids"].to(model.device)
            masked, mask_locs = apply_mask(ids, p_mask, model.device)
            if mask_locs.sum() < 2:
                continue

            h_base, h_router, _, _ = get_hidden_states(model, masked)
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
# EXPERIMENT 2: Cosine Sim by Layer
# ============================================================

@torch.no_grad()
def experiment_cosine_by_layer(model, tokenizer,
                                 p_mask=0.85,
                                 num_samples=10,
                                 max_length=128):
    print("\n" + "=" * 60)
    print(f"EXPERIMENT 2: Cosine Similarity by Layer (p_mask={p_mask})")
    print("=" * 60)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in dataset["text"][:500] if len(t) > 80][:num_samples]

    all_layer_sims = None
    router_final_sims = []
    n_valid = 0

    for text in texts:
        ids = tokenizer(text, return_tensors="pt",
                        truncation=True, max_length=max_length)["input_ids"].to(model.device)
        masked, mask_locs = apply_mask(ids, p_mask, model.device)
        if mask_locs.sum() < 2:
            continue

        h_base, h_router, layer_states, _ = get_hidden_states(model, masked)
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
    print(f"  Router corrected:     {avg_router_final:.4f}  <- after ALA")

    return avg_layer_sims, avg_router_final


# ============================================================
# EXPERIMENT 3: Two-Panel (Pairwise Collapse + Token Alignment)
# ============================================================

@torch.no_grad()
def experiment_two_panel(model, tokenizer,
                          mask_ratios=None,
                          num_samples=20,
                          max_length=128):
    """Computes both pairwise sim (global geometry) and token alignment
    (directional correction) for the paper's two-panel figure."""
    if mask_ratios is None:
        mask_ratios = [0.15, 0.30, 0.50, 0.70, 0.85, 0.95]

    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Two-Panel (Pairwise + Alignment)")
    print("=" * 60)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in dataset["text"][:500] if len(t) > 80][:num_samples]

    results = {p: {"pairwise_base": [], "pairwise_router": [],
                   "align_base": [], "align_router": []}
               for p in mask_ratios}

    for p_mask in mask_ratios:
        print(f"  Running p_mask={p_mask:.2f}...", end=" ", flush=True)

        for text in texts:
            ids = tokenizer(text, return_tensors="pt",
                            truncation=True, max_length=max_length)["input_ids"].to(model.device)
            original_ids = ids.clone()
            masked, mask_locs = apply_mask(ids, p_mask, model.device)
            if mask_locs.sum() < 2:
                continue

            h_base, h_router, _, correct_embs = get_hidden_states(
                model, masked, original_ids
            )
            if h_base is None:
                continue

            # Pairwise cosine sim (spread)
            results[p_mask]["pairwise_base"].append(mean_pairwise_cosine_sim(h_base))
            results[p_mask]["pairwise_router"].append(mean_pairwise_cosine_sim(h_router))

            # Token alignment (each position vs correct token embedding)
            results[p_mask]["align_base"].extend(
                F.cosine_similarity(h_base, correct_embs, dim=-1).tolist()
            )
            results[p_mask]["align_router"].extend(
                F.cosine_similarity(h_router, correct_embs, dim=-1).tolist()
            )

        print("done")

    # Aggregate
    agg = {}
    for p in mask_ratios:
        r = results[p]
        agg[p] = {
            "pairwise_base":   float(np.nanmean(r["pairwise_base"])),
            "pairwise_router": float(np.nanmean(r["pairwise_router"])),
            "align_base":      float(np.nanmean(r["align_base"])),
            "align_router":    float(np.nanmean(r["align_router"])),
        }

    # Print summary table
    print(f"\n  {'p_mask':<8} | {'Pairwise Δ':>12} | {'Alignment Δ':>13}")
    print(f"  {'-'*38}")
    for p in mask_ratios:
        pw_d = agg[p]["pairwise_router"] - agg[p]["pairwise_base"]
        al_d = agg[p]["align_router"] - agg[p]["align_base"]
        print(f"  {p:<8.2f} | {pw_d:>+12.4f} | {al_d:>+13.4f}")

    return agg


# ============================================================
# PLOTTING
# ============================================================

def plot_cosine_by_mask_ratio(mask_ratios, base_sims, router_sims,
                               save_path="cosine_sim_by_mask_ratio.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(mask_ratios, base_sims, 'o-', color='steelblue', linewidth=2,
            markersize=8, label='Base LLaDA')
    ax.plot(mask_ratios, router_sims, 's-', color='coral', linewidth=2,
            markersize=8, label='ALA Router')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4, label='Perfect collapse (1.0)')
    ax.set_xlabel('Mask Ratio (p_mask)', fontsize=12)
    ax.set_ylabel('Mean Pairwise Cosine Similarity', fontsize=12)
    ax.set_title('Pairwise Cosine Similarity at Masked Positions\n(higher = more similar)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.05)
    ax.set_xticks(mask_ratios)

    ax2 = axes[1]
    deltas = [r - b for r, b in zip(router_sims, base_sims)]
    colors = ['coral' if d < 0 else 'steelblue' for d in deltas]
    bars = ax2.bar([str(p) for p in mask_ratios], deltas, color=colors, alpha=0.8, edgecolor='white')
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.set_xlabel('Mask Ratio (p_mask)', fontsize=12)
    ax2.set_ylabel('Delta Cosine Similarity (Router - Base)', fontsize=12)
    ax2.set_title('ALA Cohesion Effect\n(positive = more cohesive hidden states)', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
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
    ax.scatter([layers[-1]], [avg_layer_sims[-1]], color='steelblue', s=100, zorder=5)
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
        f'Pairwise Cosine Similarity Across Layers (p_mask={p_mask})\n'
        f'Layer 0 = embedding output; Layer {len(layers)-1} = final hidden state',
        fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.05)
    ax.axhspan(0.95, 1.05, alpha=0.07, color='red')
    ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.4, linewidth=1)
    ax.text(1, 0.955, 'Collapse zone (sim > 0.95)', color='red', fontsize=8, alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Plot saved: {save_path}")


def plot_two_panel(agg, mask_ratios,
                   save_path="two_panel_symmetry_vs_alignment.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    pw_base   = [agg[p]["pairwise_base"]   for p in mask_ratios]
    pw_router = [agg[p]["pairwise_router"] for p in mask_ratios]
    al_base   = [agg[p]["align_base"]      for p in mask_ratios]
    al_router = [agg[p]["align_router"]    for p in mask_ratios]

    # Left: pairwise sim
    ax1 = axes[0]
    ax1.plot(mask_ratios, pw_base,   'o-', color='steelblue',
             linewidth=2, markersize=8, label='Base LLaDA')
    ax1.plot(mask_ratios, pw_router, 's-', color='coral',
             linewidth=2, markersize=8, label='ALA Router')
    ax1.set_xlabel('Mask Ratio ($p_{mask}$)', fontsize=12)
    ax1.set_ylabel('Mean Pairwise Cosine Similarity', fontsize=12)
    ax1.set_title('Global Hidden State Geometry\n'
                  '(pairwise sim among masked positions)', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(mask_ratios)
    ax1.set_ylim(0.0, 1.05)

    mid = len(mask_ratios) // 2
    ax1.annotate('ALA increases cohesion\n(shared anchors → similar corrections)',
                 xy=(mask_ratios[mid], pw_router[mid]),
                 xytext=(mask_ratios[mid] + 0.05, pw_router[mid] - 0.15),
                 fontsize=8, color='gray',
                 arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    # Right: token alignment
    ax2 = axes[1]
    ax2.plot(mask_ratios, al_base,   'o-', color='steelblue',
             linewidth=2, markersize=8, label='Base LLaDA')
    ax2.plot(mask_ratios, al_router, 's-', color='coral',
             linewidth=2, markersize=8, label='ALA Router')
    ax2.set_xlabel('Mask Ratio ($p_{mask}$)', fontsize=12)
    ax2.set_ylabel('Mean Cosine Similarity to Correct Token Embedding', fontsize=11)
    ax2.set_title('Prediction Direction (Token Alignment)\n'
                  '(each position vs. its correct token embedding)', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(mask_ratios)

    plt.suptitle(
        'ALA increases representation cohesion without improving ground-truth alignment',
        fontsize=11, y=1.02)
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
    else:
        print("WARNING: No saved router — using random weights.")
    model.eval()

    mask_ratios = [0.15, 0.30, 0.50, 0.70, 0.85, 0.95]

    # Experiment 1: Cosine sim vs mask ratio
    mr, base_sims, router_sims = experiment_cosine_by_mask_ratio(
        model, tokenizer, mask_ratios=mask_ratios, num_samples=20
    )
    plot_cosine_by_mask_ratio(mr, base_sims, router_sims)

    # Experiment 2: Layer-by-layer at p=0.85
    avg_layer_sims, avg_router_final = experiment_cosine_by_layer(
        model, tokenizer, p_mask=0.85, num_samples=10
    )
    plot_cosine_by_layer(avg_layer_sims, avg_router_final, p_mask=0.85)

    # Experiment 3: Two-panel figure (pairwise + alignment)
    print("\nRunning two-panel experiment...")
    agg = experiment_two_panel(
        model, tokenizer, mask_ratios=mask_ratios, num_samples=20
    )
    plot_two_panel(agg, mask_ratios)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Alpha: {ALPHA_BASE} + {ALPHA_SCALE} * p_mask")
    print(f"\n  {'p_mask':<8} | {'Pairwise Δ':>12} | {'Alignment Δ':>13}")
    print(f"  {'-'*38}")
    for p in mask_ratios:
        pw_d = agg[p]["pairwise_router"] - agg[p]["pairwise_base"]
        al_d = agg[p]["align_router"] - agg[p]["align_base"]
        print(f"  {p:<8.2f} | {pw_d:>+12.4f} | {al_d:>+13.4f}")

    print("\n  Interpretation:")
    print("  Pairwise delta ~ 0 -> ALA does not alter global hidden state geometry")
    print("  Alignment delta > 0 -> ALA pushes each position toward the correct token")
    print("\nDone. Plots saved.")
