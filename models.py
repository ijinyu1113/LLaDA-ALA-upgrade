"""
ALA (Adaptive Layer Alignment) Model Definitions
=================================================
Shared architecture for training and inference.

Two router variants:
  - AMIPRouterTrain:     forward(h_anchor, h_mask) — pre-paired [N, d] inputs
  - AMIPRouterInference: forward(h_L, mask_indices, unmasked_indices) — full
                         sequence [B, L, d] with attention-weighted aggregation

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
ALPHA_SCALE = 0.25
MASK_TOKEN_ID = 126336


def _make_experts(d_model, K):
    """Expert architecture shared by both router variants."""
    return nn.ModuleList([
        nn.Sequential(
            nn.Linear(d_model * 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model),
        ) for _ in range(K)
    ])


# ============================================================
# TRAINING ROUTER
# ============================================================
class AMIPRouterTrain(nn.Module):
    """Router for training: takes pre-paired (anchor, mask) vectors.

    Training creates explicit (mask, unmasked_anchor) pairs via vectorized
    pair finding, so forward receives pre-paired [N, d] tensors directly.

    Args:
        h_anchor: [N, d_model] hidden states at unmasked anchor positions
        h_mask:   [N, d_model] hidden states at masked positions
    Returns:
        [N, d_model] delta corrections
    """
    def __init__(self, d_model=4096, K=8):
        super().__init__()
        self.routing_net = nn.Linear(d_model, K)
        self.experts = _make_experts(d_model, K)

    def forward(self, h_anchor, h_mask):
        weights = F.softmax(self.routing_net(h_mask), dim=-1)       # [N, K]
        conditioned = torch.cat([h_anchor, h_mask], dim=-1)          # [N, 2d]
        expert_out = torch.stack(
            [e(conditioned) for e in self.experts], dim=0
        )                                                            # [K, N, d]
        weighted = expert_out * weights.t().unsqueeze(-1)            # [K, N, d]
        return weighted.sum(dim=0)                                   # [N, d]


# ============================================================
# INFERENCE ROUTER
# ============================================================
class AMIPRouterInference(nn.Module):
    """Router for inference: one mask token can have multiple unmasked anchors.

    For each masked position, finds adjacent unmasked tokens within range_r,
    computes per-pair delta via MoE, then combines via scaled dot-product
    similarity weights (attention-like relevance aggregation).

    Args:
        h_L:             [B, L, d_model] last-layer hidden states
        mask_indices:    list of B tensors, each [N_mask_b] (positions of mask tokens)
        unmasked_indices: list of B tensors, each [N_unmask_b]
        range_r:         max distance to look for anchors (default 5)
    Returns:
        [B, L, d_model] delta corrections (zero at non-masked positions)
    """
    def __init__(self, d_model=4096, K=8):
        super().__init__()
        self.routing_net = nn.Linear(d_model, K)
        self.experts = _make_experts(d_model, K)

    def forward(self, h_L, mask_indices, unmasked_indices, range_r=5):
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

                # Attention-weighted aggregation across anchors
                scores = (h_anchors * h_mask).sum(dim=-1) / (d_model ** 0.5)
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

        blended = (1 - alpha) * h_L + alpha * delta
        logits = self.base_model.model.transformer.ff_out(blended)
        if self.base_model.model.config.scale_logits:
            logits *= 1.0 / math.sqrt(self.base_model.model.config.d_model)

        return type('Obj', (object,), {'logits': logits})()

    def base_logits(self, input_ids):
        return self.base_model(input_ids).logits
