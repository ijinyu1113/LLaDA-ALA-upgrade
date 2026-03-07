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
# Special tokens whose logits the router must never modify.
# Without this, the router's delta suppresses termination tokens,
# causing infinite repetition loops (e.g. "#### 18 #### 18 ####...")
SPECIAL_TOKEN_IDS = [126081, 126348]  # <|endoftext|>, <|eot_id|>


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
        # Decide how much each of the K=8 experts should contribute for each pair
        weights = F.softmax(self.routing_net(h_mask), dim=-1)       # [N, K] — soft routing weights per pair

        # Concatenate anchor + mask hidden states as context for experts
        conditioned = torch.cat([h_anchor, h_mask], dim=-1)          # [N, 2d]

        # Run ALL experts on ALL pairs (soft MoE: every expert processes every pair)
        expert_out = torch.stack(
            [e(conditioned) for e in self.experts], dim=0             # list of K tensors [N, d] -> stack
        )                                                            # [K, N, d]

        # Multiply each expert's output by its routing weight:
        # weights.t() transposes [N, K] -> [K, N]
        # .unsqueeze(-1) adds a dim: [K, N] -> [K, N, 1] so it broadcasts with [K, N, d]
        weighted = expert_out * weights.t().unsqueeze(-1)            # [K, N, d]

        # Sum across experts (dim=0) to get one combined delta per pair
        delta = weighted.sum(dim=0)                                  # [N, d]

        # --- Learned Relevance Gate ---
        # "Is this anchor actually useful for predicting the masked token?"
        # Projects both into a shared low-dim space, computes scaled dot product
        q = self.q_proj(h_mask)                                      # [N, d_proj]
        k = self.k_proj(h_anchor)                                    # [N, d_proj]
        # (q * k) = element-wise multiply [N, d_proj], .sum(dim=-1) = dot product -> [N]
        # keepdim=True makes it [N, 1] so it broadcasts with delta [N, d]
        # sigmoid outputs 0-1: ~0 means "irrelevant anchor", ~1 means "useful anchor"
        relevance = torch.sigmoid(
            (q * k).sum(dim=-1, keepdim=True) / (self.d_proj ** 0.5)
        )                                                            # [N, 1]
        return delta * relevance  # gate: irrelevant anchors produce near-zero corrections


# ============================================================
# INFERENCE ROUTER
# ============================================================
class AMIPRouterInference(nn.Module):
    """Router for inference: one mask token can have multiple unmasked anchors.

    For each masked position, finds adjacent unmasked tokens within range_r,
    computes per-pair delta via MoE, then combines via learned Q/K attention
    weights (trained relevance, not raw dot product).

    Args:
        h_L:             [B, L, d_model] last-layer hidden states
        mask_indices:    list of B tensors, each [N_mask_b]
        unmasked_indices: list of B tensors, each [N_unmask_b]
        range_r:         max distance to look for anchors (default RANGE_R)
    Returns:
        [B, L, d_model] delta corrections (zero at non-masked positions)
    """
    def __init__(self, d_model=4096, K=8):
        super().__init__()
        self.d_proj = d_model // 8
        self.routing_net = nn.Linear(d_model, K)
        self.experts = _make_experts(d_model, K)
        self.q_proj = nn.Linear(d_model, self.d_proj)
        self.k_proj = nn.Linear(d_model, self.d_proj)

    def forward(self, h_L, mask_indices, unmasked_indices, range_r=RANGE_R):
        delta_h = torch.zeros_like(h_L)  # [B, L, d] — output starts as all zeros
        bsz, seq_len, d_model = h_L.shape

        for b in range(bsz):  # process each sample in the batch
            m_idx, u_idx = mask_indices[b], unmasked_indices[b]
            if not isinstance(u_idx, torch.Tensor):
                u_idx = torch.tensor(u_idx, device=h_L.device)

            for a in m_idx:  # for each masked position in this sample
                # Find unmasked neighbors within range_r positions
                diff = (u_idx - a).abs()
                adj = u_idx[(diff > 0) & (diff <= range_r)]  # boolean filter: nearby + unmasked
                if len(adj) == 0:
                    continue  # no unmasked neighbors -> skip (delta stays zero)

                N = len(adj)  # number of anchor neighbors for this mask token
                # a:a+1 (slice) keeps the dim -> [1, d]; a (index) would give [d]
                # We need [1, d] so it broadcasts correctly with [N, d] below
                h_mask = h_L[b, a:a+1, :]                             # [1, d]
                h_anchors = h_L[b, adj, :]                             # [N, d] — fancy index with tensor of positions

                # --- MoE routing (same computation as training router) ---
                # Route using the mask token (all anchors share the same routing weights)
                weights = F.softmax(self.routing_net(h_mask), dim=-1)   # [1, K]
                # expand: repeat [1, d] -> [N, d] without copying memory, then concat with anchors
                conditioned = torch.cat(
                    [h_anchors, h_mask.expand(N, -1)], dim=-1
                )                                                       # [N, 2d]
                # Weighted sum of all experts' outputs for each anchor
                # weights[:, i:i+1] is [1, 1], expert(conditioned) is [N, d]
                # Broadcasting: [1, 1] * [N, d] -> [N, d], then sum across all experts
                pair_deltas = sum(
                    weights[:, i:i+1] * expert(conditioned)
                    for i, expert in enumerate(self.experts)
                )                                                       # [N, d]

                # --- Learned attention aggregation across anchors ---
                # Unlike training (sigmoid per pair), inference uses softmax to RANK anchors:
                # "Which of these N anchors is most useful?" not "Is this one anchor useful?"
                q = self.q_proj(h_mask)                                 # [1, d_proj]
                k = self.k_proj(h_anchors)                              # [N, d_proj]
                # q * k broadcasts [1, d_proj] * [N, d_proj] -> [N, d_proj], then sum -> [N]
                scores = (q * k).sum(dim=-1) / (self.d_proj ** 0.5)    # [N] — one score per anchor
                combine_w = F.softmax(scores, dim=0)                    # [N] — normalized weights (sum to 1)
                # Weighted combination: [N, 1] * [N, d] -> [N, d], sum over anchors -> [d]
                delta_h[b, a] = (combine_w.unsqueeze(-1) * pair_deltas).sum(0)

        return delta_h


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
            alpha = ALPHA_BASE + ALPHA_SCALE * p_mask  # with ALPHA_SCALE=0, always 0.1

        # Residual blending: original hidden state + small correction
        blended = h_L + alpha * delta
        # Project blended hidden states to vocabulary logits using base model's output head (frozen)
        logits = self.base_model.model.transformer.ff_out(blended)
        if self.base_model.model.config.scale_logits:
            logits *= 1.0 / math.sqrt(self.base_model.model.config.d_model)

        # Restore base model's logits for special tokens (endoftext, eot_id).
        # The base model forward pass already computed these — zero extra cost.
        # This prevents the router delta from suppressing termination signals.
        base_logits = outputs.logits
        logits[:, :, SPECIAL_TOKEN_IDS] = base_logits[:, :, SPECIAL_TOKEN_IDS]

        # Create a simple object with .logits attribute (duck-typing to match HuggingFace output format)
        return type('Obj', (object,), {'logits': logits})()

    def base_logits(self, input_ids):
        """Run base model only (no router) — used as baseline comparison."""
        return self.base_model(input_ids).logits
