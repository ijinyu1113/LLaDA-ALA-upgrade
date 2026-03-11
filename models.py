"""
ALA (Adaptive Layer Alignment) Model Definitions
=================================================
Shared architecture for training and inference.

Two router variants:
  - AMIPRouterTrain:     forward(h_anchor, h_mask) — pre-paired [N, d] inputs
  - AMIPRouterInference: forward(h_L, mask_indices, unmasked_indices) — full
                         sequence [B, L, d] with learned attention aggregation

Alpha schedule: alpha = ALPHA_BASE + ALPHA_SCALE * p_mask
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================
# SHARED CONSTANTS
# ============================================================
ALPHA_BASE = 0.05
ALPHA_SCALE = 0.0  # flat alpha — adaptive schedule hurt generation quality
MASK_TOKEN_ID = 126336
RANGE_R = 10  # how far to look for unmasked anchors


def _make_experts(d_model, K):
    """Expert architecture shared by both router variants.

    Bottleneck: 2d -> d/2 -> d (wider than d/4 to preserve fine-grained token distinctions).
    Last layer zero-init so router starts as identity (no correction).
    """
    experts = nn.ModuleList()
    for _ in range(K):
        layers = nn.Sequential(
            # Input is concat(h_anchor, h_mask) = 2*d_model dimensions
            # Compress to d_model//2 (bottleneck forces the expert to learn a compact representation)
            nn.Linear(d_model * 2, d_model // 2),
            nn.GELU(),
            # Expand back to d_model (same size as hidden states, so we can add it as a correction)
            nn.Linear(d_model // 2, d_model),
        )
        # Zero-init the output layer: at init, every expert outputs all zeros
        # This means the router starts as identity (no correction), then gradually learns corrections
        nn.init.zeros_(layers[2].weight)
        nn.init.zeros_(layers[2].bias)
        experts.append(layers)
    return experts


# ============================================================
# TRAINING ROUTER
# ============================================================
class AMIPRouterTrain(nn.Module):
    """Router for training: takes pre-paired (anchor, mask) vectors.

    Each pair gets a learned relevance gate (sigmoid of q·k) that scales
    the MoE delta — irrelevant anchors produce near-zero corrections.

    Args:
        h_anchor: [N, d_model] hidden states at unmasked anchor positions
        h_mask:   [N, d_model] hidden states at masked positions
    Returns:
        [N, d_model] relevance-gated delta corrections
    """
    def __init__(self, d_model=4096, K=8):
        super().__init__()
        self.d_proj = d_model // 8  # projection dim for Q/K attention (4096 -> 512)
        self.routing_net = nn.Linear(d_model, K)  # maps hidden state -> K expert weights
        self.experts = _make_experts(d_model, K)
        self.q_proj = nn.Linear(d_model, self.d_proj)  # query projection (from mask token)
        self.k_proj = nn.Linear(d_model, self.d_proj)  # key projection (from anchor token)

    def forward(self, h_anchor, h_mask):
        # --- Mixture of Experts (MoE) ---
        weights = F.softmax(self.routing_net(h_mask), dim=-1)       # [N, K]
        conditioned = torch.cat([h_anchor, h_mask], dim=-1)          # [N, 2d]
        expert_out = torch.stack(
            [e(conditioned) for e in self.experts], dim=0             # [K, N, d]
        )
        weighted = expert_out * weights.t().unsqueeze(-1)            # [K, N, d]
        delta = weighted.sum(dim=0)                                  # [N, d]

        # --- Learned Relevance Gate (per anchor-mask pair) ---
        q = self.q_proj(h_mask)                                      # [N, d_proj]
        k = self.k_proj(h_anchor)                                    # [N, d_proj]
        relevance = torch.sigmoid(
            (q * k).sum(dim=-1, keepdim=True) / (self.d_proj ** 0.5)
        )                                                            # [N, 1]

        return delta * relevance


# ============================================================
# INFERENCE ROUTER
# ============================================================
class AMIPRouterInference(nn.Module):
    """Router for inference: one mask token can have multiple unmasked anchors.

    At inference, each masked position may have several nearby unmasked tokens
    (anchors). For each (mask, anchor) pair, the MoE experts produce a correction
    vector (d=4096 dims, NOT a scalar). Then learned Q/K attention weights combine
    the correction vectors from all anchors into one final correction per mask position.

    Notation:
        B = batch size (usually 1 at inference)
        L = sequence length (e.g. 512 tokens)
        d = d_model = 4096 (hidden state dimension)
        M = number of masked positions in one sequence
        U = number of unmasked positions in one sequence
        P = number of (mask, anchor) pairs after adjacency filtering
        K = number of MoE experts (8)

    Args:
        h_L:             [B, L, d] last-layer hidden states from frozen base model
        mask_indices:    list of B tensors, each [M_b] — positions of [MASK] tokens
        unmasked_indices: list of B tensors, each [U_b] — positions of non-[MASK] tokens
        range_r:         max distance to look for anchors (default 10)
    Returns:
        [B, L, d] correction vectors (zero vectors at non-masked positions)
    """
    def __init__(self, d_model=4096, K=8):
        super().__init__()
        self.d_proj = d_model // 8                   # 4096 -> 512 projection for Q/K attention
        self.routing_net = nn.Linear(d_model, K)     # maps h_mask -> K expert weights
        self.experts = _make_experts(d_model, K)     # K experts, each: [2d] -> [d/2] -> [d]
        self.q_proj = nn.Linear(d_model, self.d_proj)  # query proj (from mask token)
        self.k_proj = nn.Linear(d_model, self.d_proj)  # key proj (from anchor token)

    def forward(self, h_L, mask_indices, unmasked_indices, range_r=RANGE_R):
        # Output: one correction vector per position. Non-masked positions stay as zero vectors.
        delta_h = torch.zeros_like(h_L)              # [B, L, d] — all zeros initially
        bsz, seq_len, d_model = h_L.shape

        for b in range(bsz):                         # loop over batch (usually just 1)
            m_idx = mask_indices[b]                   # [M] — sequence positions of [MASK] tokens
            u_idx = unmasked_indices[b]               # [U] — sequence positions of unmasked tokens
            if not isinstance(u_idx, torch.Tensor):
                u_idx = torch.tensor(u_idx, device=h_L.device)
            if len(m_idx) == 0 or len(u_idx) == 0:
                continue

            # ---- Step 1: Find all (mask, anchor) pairs within range_r ----
            # Compute |position_mask - position_anchor| for every mask × anchor combination
            dists = (m_idx.unsqueeze(1).float()       # [M, 1] — each mask position
                     - u_idx.unsqueeze(0).float()     # [1, U] — each anchor position
                     ).abs()                          # [M, U] — absolute distances
            # Keep pairs where anchor is within range but not at the same position
            adj_mask = (dists > 0) & (dists <= range_r)  # [M, U] — True where pair is valid

            # Extract flat list of valid pairs: pair_m indexes into m_idx, pair_u into u_idx
            # Example: if mask pos 5 has anchors at pos 3,7 → pair_m=[0,0], pair_u=[idx_of_3, idx_of_7]
            pair_m, pair_u = adj_mask.nonzero(as_tuple=True)  # each [P]
            if len(pair_m) == 0:
                continue

            # Gather the hidden state vectors for all P pairs at once
            h_masks = h_L[b, m_idx[pair_m]]           # [P, d] — hidden state at each mask position
            h_anchors = h_L[b, u_idx[pair_u]]         # [P, d] — hidden state at each anchor position

            # ---- Step 2: MoE produces one correction vector per pair ----
            # Each mask token decides how to weight the K=8 experts
            weights = F.softmax(self.routing_net(h_masks), dim=-1)  # [P, K] — soft routing weights
            # Experts take concat(anchor, mask) as input — context from both sides
            conditioned = torch.cat([h_anchors, h_masks], dim=-1)   # [P, 2d] = [P, 8192]
            # Weighted sum of all experts' outputs: each expert produces a [P, d] correction vector
            pair_deltas = sum(
                weights[:, i:i+1] * expert(conditioned)  # [P, 1] * [P, d] → [P, d]
                for i, expert in enumerate(self.experts)
            )  # [P, d] — one correction vector per (mask, anchor) pair

            # ---- Step 3: Attention scores to combine multiple anchors per mask ----
            # "How much should each anchor's correction contribute to this mask position?"
            q = self.q_proj(h_masks)                  # [P, 512] — query from mask
            k = self.k_proj(h_anchors)                # [P, 512] — key from anchor
            scores = (q * k).sum(dim=-1) / (self.d_proj ** 0.5)  # [P] — scaled dot-product

            # ---- Step 4: Per-mask-position softmax via scatter operations ----
            # We need softmax grouped by mask position (each mask has different anchors)
            # This is equivalent to: for each mask, softmax over its anchors' scores
            # But done without a Python loop using scatter operations
            num_masks = len(m_idx)
            # Find max score per mask position (for numerical stability)
            max_scores = torch.full((num_masks,), -1e9, device=scores.device, dtype=scores.dtype)
            max_scores.scatter_reduce_(0, pair_m, scores, reduce='amax')  # [M] — max per group
            # Stable exp: subtract per-group max before exponentiating
            exp_scores = torch.exp(scores - max_scores[pair_m])           # [P]
            # Sum of exp per mask position (denominator of softmax)
            sum_exp = torch.zeros(num_masks, device=scores.device, dtype=scores.dtype)
            sum_exp.scatter_add_(0, pair_m, exp_scores)                   # [M]
            # Normalized attention weights: softmax(score) per mask position
            combine_w = exp_scores / sum_exp[pair_m].clamp(min=1e-8)      # [P] — sums to 1 per mask

            # ---- Step 5: Aggregate correction vectors back to sequence positions ----
            # Weight each pair's correction vector by its attention weight
            weighted_deltas = combine_w.unsqueeze(-1) * pair_deltas       # [P, d]
            # Scatter-add: accumulate weighted corrections to the right sequence positions
            # Multiple pairs targeting the same mask position get summed together
            delta_h[b].scatter_add_(
                0,                                    # scatter along dim 0 (sequence positions)
                m_idx[pair_m].unsqueeze(-1).expand_as(weighted_deltas),  # [P, d] — target positions
                weighted_deltas                       # [P, d] — values to scatter
            )  # delta_h[b] is [L, d] — now has correction vectors at masked positions

        return delta_h  # [B, L, d] — correction vectors, added to h_L as: h_blended = h_L + α * delta_h


# ============================================================
# ALALLaDA WRAPPER
# ============================================================
class ALALLaDA(nn.Module):
    """Wraps frozen base LLaDA + ALA router.

    Auto-computes alpha from current mask ratio unless overridden.
    prompt_length excludes prompt tokens from p_mask computation
    (they are never masked, so including them deflates p_mask).
    """
    def __init__(self, base_model, router=None):
        super().__init__()
        self.base_model = base_model
        self.router = router if router is not None else AMIPRouterInference()

    @property
    def device(self):
        return next(self.base_model.parameters()).device

    def forward(self, input_ids, attention_mask=None, alpha=None,
                prompt_length=0):
        # Run frozen base model, get hidden states from every layer
        outputs = self.base_model(
            input_ids, attention_mask=attention_mask,
            output_hidden_states=True,  # returns tuple of hidden states from all layers
        )
        h_L = outputs.hidden_states[-1].to(torch.bfloat16)  # last layer: [B, L, d]

        # Find masked vs unmasked positions per batch element
        # torch.where returns indices where condition is True
        m_idx = [torch.where(row == MASK_TOKEN_ID)[0] for row in input_ids]  # list of B index tensors
        u_idx = [torch.where(row != MASK_TOKEN_ID)[0] for row in input_ids]
        delta = self.router(h_L, m_idx, u_idx)  # [B, L, d] — corrections (zero at non-mask positions)

        if alpha is None:
            # Auto-compute alpha from current mask ratio in the generation region only
            # (prompt tokens are never masked, including them would make p_mask artificially low)
            gen_region = input_ids[:, prompt_length:]
            p_mask = (gen_region == MASK_TOKEN_ID).float().mean().item()
            alpha = ALPHA_BASE + ALPHA_SCALE * p_mask  # with ALPHA_SCALE=0, always 0.05

        # Residual blending: original hidden state + small correction
        blended = h_L + alpha * delta
        # Project blended hidden states to vocabulary logits using base model's output head (frozen)
        logits = self.base_model.model.transformer.ff_out(blended)
        if self.base_model.model.config.scale_logits:
            logits *= 1.0 / math.sqrt(self.base_model.model.config.d_model)

        # Create a simple object with .logits attribute (duck-typing to match HuggingFace output format)
        return type('Obj', (object,), {'logits': logits})()

    def base_logits(self, input_ids):
        """Run base model only (no router) — used as baseline comparison."""
        return self.base_model(input_ids).logits
