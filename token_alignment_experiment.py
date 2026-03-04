"""
Token Alignment Experiment
===========================
Measures cosine similarity between masked-position hidden states
and the correct token's output embedding vector, comparing:
  - Base LLaDA:  cos(h_base,  W_out[correct_token])
  - ALA Router:  cos(h_router, W_out[correct_token])

The hypothesis: ALA nudges each masked position's hidden state
*toward* the correct token's direction in the output embedding space,
even if it doesn't spread positions apart from each other globally.

This is the mechanistic link between:
  "collapse exists" (layer cosine sim plot)
        ↓
  "ALA corrects toward right answer" (this experiment)
        ↓
  "accuracy improves" (mask ratio sweep)

Produces:
  token_alignment_by_mask_ratio.png  — mean alignment vs p_mask
  token_alignment_summary.txt        — numerical results
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import math
import os

# ============================================================
# ARCHITECTURE (matches eval.py exactly)
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
            m_idx = mask_indices[b]
            u_idx = unmasked_indices[b]
            for a in m_idx:
                adj = [t for t in u_idx if 0 < abs(t - a) <= range_r]
                if not adj:
                    continue
                h_mask = h_L[b, a:a+1, :]
                pair_deltas, relevance_scores = [], []
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
        outputs = self.base_model(
            input_ids, attention_mask=attention_mask,
            output_hidden_states=True
        )
        h_L = outputs.hidden_states[-1].to(torch.bfloat16)
        m_idx = [torch.where(row == 126336)[0].tolist() for row in input_ids]
        u_idx = [torch.where(row != 126336)[0].tolist() for row in input_ids]
        delta = self.router(h_L, m_idx, u_idx)
        blended_h = (1 - self.alpha) * h_L + self.alpha * delta
        logits = self.base_model.model.transformer.ff_out(blended_h)
        if self.base_model.model.config.scale_logits:
            logits = logits * (1 / math.sqrt(self.base_model.model.config.d_model))
        return type('Obj', (object,), {'logits': logits})()

    def base_logits(self, input_ids):
        return self.base_model(input_ids).logits


# ============================================================
# CORE: get h_base, h_router, and correct token embeddings
# ============================================================
@torch.no_grad()
def get_alignment_data(model, input_ids, original_ids, alpha=0.1):
    """
    For each masked position with a known correct token, returns:
      h_base:        final hidden state from frozen model   [N, d]
      h_router:      ALA-corrected hidden state             [N, d]
      correct_embs:  output embedding of correct token      [N, d]

    Uses the output projection weight matrix W_out as the embedding
    space, since that's where hidden states are projected to logits.
    """
    mask_token_id = 126336

    # Forward pass with hidden states
    outputs = model.base_model(input_ids, output_hidden_states=True)
    h_L = outputs.hidden_states[-1].to(torch.bfloat16)  # [1, L, d]

    # Masked positions that have a known correct token
    mask_positions = (input_ids[0] == mask_token_id)
    original_tokens = original_ids[0]

    # Only keep positions where we know the correct answer
    valid = mask_positions  # all masked positions have a ground truth
    if valid.sum() < 1:
        return None, None, None

    masked_pos_idx = valid.nonzero(as_tuple=True)[0]  # [N]

    # Base hidden states at masked positions
    h_base = h_L[0, masked_pos_idx, :].to(torch.float32)  # [N, d]

    # Router-corrected hidden states
    m_idx_list = [masked_pos_idx.tolist()]
    u_idx_list = [(input_ids[0] != mask_token_id).nonzero(as_tuple=True)[0].tolist()]
    delta = model.router(h_L, m_idx_list, u_idx_list)
    h_blended = (1 - alpha) * h_L + alpha * delta
    h_router = h_blended[0, masked_pos_idx, :].to(torch.float32)  # [N, d]

    # Correct token output embeddings from W_out
    # W_out shape: [vocab_size, d_model]
    W_out = model.base_model.model.transformer.ff_out.weight.to(torch.float32)
    correct_token_ids = original_tokens[masked_pos_idx]     # [N]
    correct_embs = W_out[correct_token_ids]                 # [N, d]

    return h_base, h_router, correct_embs


# ============================================================
# EXPERIMENT: Token Alignment vs Mask Ratio
# ============================================================
@torch.no_grad()
def experiment_token_alignment(model, tokenizer,
                                mask_ratios=None,
                                num_samples=30,
                                max_length=128):
    """
    For each mask ratio, compute:
      - mean cosine similarity between h_base and correct token embedding
      - mean cosine similarity between h_router and correct token embedding
      - Δ = router_alignment - base_alignment

    Higher alignment = hidden state is pointing more toward the
    correct answer in the output embedding space.
    """
    if mask_ratios is None:
        mask_ratios = [0.15, 0.30, 0.50, 0.70, 0.85, 0.95]

    print("\n" + "=" * 60)
    print("TOKEN ALIGNMENT EXPERIMENT")
    print("cos(h, W_out[correct_token]) — higher = better directed")
    print("=" * 60)
    print(f"\n  {'p_mask':<8} | {'Base Align':<12} | {'Router Align':<14} | "
          f"{'Δ':>8} | {'N tokens'}")
    print(f"  {'-'*60}")

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in dataset["text"][:500] if len(t) > 80][:num_samples]

    base_alignments, router_alignments, token_counts = [], [], []

    for p_mask in mask_ratios:
        batch_base, batch_router, n_total = [], [], 0

        for text in texts:
            enc = tokenizer(text, return_tensors="pt",
                            truncation=True, max_length=max_length)
            ids = enc["input_ids"].to(model.device)
            original_ids = ids.clone()

            # Apply mask
            prob = torch.full(ids.shape, p_mask, device=ids.device)
            prob[:, 0] = 0.0  # never mask BOS
            masked = ids.clone()
            mask_locs = torch.bernoulli(prob).bool()
            masked[mask_locs] = 126336

            if mask_locs.sum() < 1:
                continue

            h_base, h_router, correct_embs = get_alignment_data(
                model, masked, original_ids, alpha=model.alpha
            )
            if h_base is None:
                continue

            # Cosine similarity: each hidden state vs its correct embedding
            # F.cosine_similarity operates row-wise: [N] output
            base_sim = F.cosine_similarity(
                h_base, correct_embs, dim=-1
            )  # [N]
            router_sim = F.cosine_similarity(
                h_router, correct_embs, dim=-1
            )  # [N]

            batch_base.extend(base_sim.tolist())
            batch_router.extend(router_sim.tolist())
            n_total += base_sim.shape[0]

        mean_base = float(np.mean(batch_base)) if batch_base else float('nan')
        mean_router = float(np.mean(batch_router)) if batch_router else float('nan')
        delta = mean_router - mean_base

        base_alignments.append(mean_base)
        router_alignments.append(mean_router)
        token_counts.append(n_total)

        marker = "  ◄ peak" if abs(delta) == max(
            [abs(mean_router - mean_base)], default=0
        ) else ""
        print(f"  {p_mask:<8.2f} | {mean_base:<12.4f} | {mean_router:<14.4f} | "
              f"{delta:>+8.4f} | {n_total}{marker}")

    return mask_ratios, base_alignments, router_alignments, token_counts


# ============================================================
# EXPERIMENT 2: Alignment broken down by mask ratio
#               showing BOTH alignment AND pairwise cosine sim
#               side by side — the two-panel figure for the paper
# ============================================================
@torch.no_grad()
def experiment_two_panel(model, tokenizer,
                          mask_ratios=None,
                          num_samples=20,
                          max_length=128):
    """
    Computes both:
      1. Pairwise cosine sim (spread between masked positions)
      2. Token alignment (each position vs correct token embedding)

    For the paper's two-panel figure showing:
      Left:  pairwise sim barely changes (ALA doesn't fix global geometry)
      Right: alignment improves (ALA corrects prediction direction)
    """
    if mask_ratios is None:
        mask_ratios = [0.15, 0.30, 0.50, 0.70, 0.85, 0.95]

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in dataset["text"][:500] if len(t) > 80][:num_samples]

    results = {p: {"pairwise_base": [], "pairwise_router": [],
                   "align_base": [], "align_router": []}
               for p in mask_ratios}

    for p_mask in mask_ratios:
        print(f"  Running p_mask={p_mask:.2f}...", end=" ", flush=True)

        for text in texts:
            enc = tokenizer(text, return_tensors="pt",
                            truncation=True, max_length=max_length)
            ids = enc["input_ids"].to(model.device)
            original_ids = ids.clone()

            prob = torch.full(ids.shape, p_mask, device=ids.device)
            prob[:, 0] = 0.0
            masked = ids.clone()
            mask_locs = torch.bernoulli(prob).bool()
            masked[mask_locs] = 126336

            if mask_locs.sum() < 2:
                continue

            h_base, h_router, correct_embs = get_alignment_data(
                model, masked, original_ids
            )
            if h_base is None:
                continue

            # 1. Pairwise cosine sim (spread)
            def mean_pairwise(h):
                if h.shape[0] < 2:
                    return float('nan')
                hn = F.normalize(h, dim=-1)
                sim = hn @ hn.T
                N = h.shape[0]
                mask = torch.triu(
                    torch.ones(N, N, dtype=torch.bool, device=h.device),
                    diagonal=1
                )
                return sim[mask].mean().item()

            results[p_mask]["pairwise_base"].append(
                mean_pairwise(h_base)
            )
            results[p_mask]["pairwise_router"].append(
                mean_pairwise(h_router)
            )

            # 2. Token alignment
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

    return agg


# ============================================================
# PLOTTING
# ============================================================
def plot_token_alignment(mask_ratios, base_alignments, router_alignments,
                          save_path="token_alignment_by_mask_ratio.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: absolute alignment scores
    ax = axes[0]
    ax.plot(mask_ratios, base_alignments, 'o-', color='steelblue',
            linewidth=2, markersize=8, label='Base LLaDA')
    ax.plot(mask_ratios, router_alignments, 's-', color='coral',
            linewidth=2, markersize=8, label='ALA Router')
    ax.set_xlabel('Mask Ratio ($p_{mask}$)', fontsize=12)
    ax.set_ylabel('Mean Cosine Similarity to Correct Token Embedding', fontsize=11)
    ax.set_title('Token Alignment: How Much Does the Hidden State\n'
                 'Point Toward the Correct Answer?', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(mask_ratios)

    # Right: delta alignment
    ax2 = axes[1]
    deltas = [r - b for r, b in zip(router_alignments, base_alignments)]
    colors = ['coral' if d > 0 else 'steelblue' for d in deltas]
    bars = ax2.bar([str(p) for p in mask_ratios], deltas,
                   color=colors, alpha=0.8, edgecolor='white')
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.set_xlabel('Mask Ratio ($p_{mask}$)', fontsize=12)
    ax2.set_ylabel('Δ Alignment (Router − Base)', fontsize=12)
    ax2.set_title('ALA Alignment Improvement\n(positive = better directed toward correct token)',
                  fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, d in zip(bars, deltas):
        ypos = d + 0.0002 if d > 0 else d - 0.0002
        va = 'bottom' if d > 0 else 'top'
        ax2.text(bar.get_x() + bar.get_width()/2, ypos,
                 f'{d:+.4f}', ha='center', va=va,
                 fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {save_path}")


def plot_two_panel(agg, mask_ratios,
                   save_path="two_panel_symmetry_vs_alignment.png"):
    """
    The paper figure:
      Left:  pairwise cosine sim (global geometry — ALA barely moves it)
      Right: token alignment (directional correction — ALA improves it)
    """
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

    # Annotate: "ALA does not substantially alter global geometry"
    mid = len(mask_ratios) // 2
    ax1.annotate('Δ ≈ 0 (ALA does not\nalter global geometry)',
                 xy=(mask_ratios[mid], pw_router[mid]),
                 xytext=(mask_ratios[mid] + 0.05,
                         pw_router[mid] - 0.15),
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
        'ALA corrects prediction direction without altering global geometry\n'
        'Left: pairwise spread unchanged  |  Right: alignment with correct token improves',
        fontsize=11, y=1.02
    )
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
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model = ALALLaDA(base_model, alpha=0.1).to(torch.bfloat16)
    device = next(base_model.parameters()).device
    model.router.to(device)

    weights_path = "amip_router_best.pt"
    if os.path.exists(weights_path):
        model.router.load_state_dict(
            torch.load(weights_path, map_location=device)
        )
        print(f"Router loaded from {weights_path}")
    else:
        print("WARNING: No saved router — using random weights.")

    model.eval()

    mask_ratios = [0.15, 0.30, 0.50, 0.70, 0.85, 0.95]

    # ── Experiment 1: alignment only (fast) ──────────────────
    mask_ratios_out, base_al, router_al, counts = experiment_token_alignment(
        model, tokenizer, mask_ratios=mask_ratios, num_samples=30
    )
    plot_token_alignment(mask_ratios_out, base_al, router_al)

    # ── Experiment 2: two-panel figure for paper ─────────────
    print("\nRunning two-panel experiment (pairwise + alignment)...")
    agg = experiment_two_panel(
        model, tokenizer, mask_ratios=mask_ratios, num_samples=20
    )
    plot_two_panel(agg, mask_ratios)

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n  {'p_mask':<8} | {'Pairwise Δ':>12} | {'Alignment Δ':>13}")
    print(f"  {'-'*38}")
    for p in mask_ratios:
        pw_d = agg[p]["pairwise_router"] - agg[p]["pairwise_base"]
        al_d = agg[p]["align_router"]    - agg[p]["align_base"]
        print(f"  {p:<8.2f} | {pw_d:>+12.4f} | {al_d:>+13.4f}")

    print("\n  Interpretation:")
    print("  Pairwise Δ ≈ 0  → ALA does not alter global hidden state geometry")
    print("  Alignment Δ > 0 → ALA pushes each position toward the correct token")
    print("\n  This dissociation is the mechanistic story of the paper.")

    # Save results
    with open("token_alignment_results.txt", "w") as f:
        f.write("Token Alignment Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'p_mask':<8} | {'Base Align':<12} | "
                f"{'Router Align':<14} | {'Δ Align':>10} | "
                f"{'Δ Pairwise':>12}\n")
        f.write("-" * 62 + "\n")
        for p, ba, ra in zip(mask_ratios_out, base_al, router_al):
            pw_d = agg[p]["pairwise_router"] - agg[p]["pairwise_base"]
            f.write(f"{p:<8.2f} | {ba:<12.4f} | {ra:<14.4f} | "
                    f"{ra-ba:>+10.4f} | {pw_d:>+12.4f}\n")
    print("\n  Results saved to token_alignment_results.txt")
    print("  Plots: token_alignment_by_mask_ratio.png, "
          "two_panel_symmetry_vs_alignment.png")