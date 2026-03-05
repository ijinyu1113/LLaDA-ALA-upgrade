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
ALPHA_BASE = 0.1
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
            nn.Linear(d_model * 2, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )
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
        self.d_proj = d_model // 8
        self.routing_net = nn.Linear(d_model, K)
        self.experts = _make_experts(d_model, K)
        self.q_proj = nn.Linear(d_model, self.d_proj)
        self.k_proj = nn.Linear(d_model, self.d_proj)

    def forward(self, h_anchor, h_mask):
        weights = F.softmax(self.routing_net(h_mask), dim=-1)       # [N, K]
        conditioned = torch.cat([h_anchor, h_mask], dim=-1)          # [N, 2d]
        expert_out = torch.stack(
            [e(conditioned) for e in self.experts], dim=0
        )                                                            # [K, N, d]
        weighted = expert_out * weights.t().unsqueeze(-1)            # [K, N, d]
        delta = weighted.sum(dim=0)                                  # [N, d]

        # Learned relevance gate
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
        delta_h = torch.zeros_like(h_L)
        bsz, seq_len, d_model = h_L.shape

        for b in range(bsz):
            m_idx, u_idx = mask_indices[b], unmasked_indices[b]
            if not isinstance(u_idx, torch.Tensor):
                u_idx = torch.tensor(u_idx, device=h_L.device)
            for a in m_idx:
                diff = (u_idx - a).abs()
                adj = u_idx[(diff > 0) & (diff <= range_r)]
                if len(adj) == 0:
                    continue

                N = len(adj)
                h_mask = h_L[b, a:a+1, :]                             # [1, d]
                h_anchors = h_L[b, adj, :]                             # [N, d]

                # MoE routing on the masked token
                weights = F.softmax(self.routing_net(h_mask), dim=-1)   # [1, K]
                conditioned = torch.cat(
                    [h_anchors, h_mask.expand(N, -1)], dim=-1
                )                                                       # [N, 2d]
                pair_deltas = sum(
                    weights[:, i:i+1] * expert(conditioned)
                    for i, expert in enumerate(self.experts)
                )                                                       # [N, d]

                # Learned attention aggregation across anchors
                q = self.q_proj(h_mask)                                 # [1, d_proj]
                k = self.k_proj(h_anchors)                              # [N, d_proj]
                scores = (q * k).sum(dim=-1) / (self.d_proj ** 0.5)    # [N]
                combine_w = F.softmax(scores, dim=0)                    # [N]
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
        outputs = self.base_model(
            input_ids, attention_mask=attention_mask,
            output_hidden_states=True,
        )
        h_L = outputs.hidden_states[-1].to(torch.bfloat16)
        m_idx = [torch.where(row == MASK_TOKEN_ID)[0] for row in input_ids]
        u_idx = [torch.where(row != MASK_TOKEN_ID)[0] for row in input_ids]
        delta = self.router(h_L, m_idx, u_idx)

        if alpha is None:
            gen_region = input_ids[:, prompt_length:]
            p_mask = (gen_region == MASK_TOKEN_ID).float().mean().item()
            alpha = ALPHA_BASE + ALPHA_SCALE * p_mask

        blended = h_L + alpha * delta
        logits = self.base_model.model.transformer.ff_out(blended)
        if self.base_model.model.config.scale_logits:
            logits *= 1.0 / math.sqrt(self.base_model.model.config.d_model)

        return type('Obj', (object,), {'logits': logits})()

    def base_logits(self, input_ids):
        return self.base_model(input_ids).logits
