import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import functools
print = functools.partial(print, flush=True)

# H100 optimizations
torch.set_float32_matmul_precision('high')  # TF32 tensor cores
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# =============================================================================
# 1. ARCHITECTURE: ALA Router (unchanged from your version)
# =============================================================================
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
        
    def forward(self, h_anchor, h_mask):
        weights = F.softmax(self.routing_net(h_mask), dim=-1)  # [N, K]
        conditioned_input = torch.cat([h_anchor, h_mask], dim=-1)  # [N, 2*d_model]
        
        # Stack all expert outputs at once instead of looping
        # expert_outputs: [K, N, d_model]
        expert_outputs = torch.stack([expert(conditioned_input) for expert in self.experts], dim=0)
        
        # weights: [N, K] -> [K, N, 1] for broadcasting
        weighted = expert_outputs * weights.t().unsqueeze(-1)  # [K, N, d_model]
        
        return weighted.sum(dim=0)  # [N, d_model]


# =============================================================================
# 2. VECTORIZED MASKING
# =============================================================================
def apply_random_mask(input_ids, attention_mask, p_mask, mask_token_id=126336):
    """
    Applies random masking to real tokens only.
    Returns masked input_ids and labels (-100 for non-targets).
    """
    labels = input_ids.clone()
    
    # Mask only where attention_mask == 1 (real tokens)
    probability_matrix = torch.full(labels.shape, p_mask, device=input_ids.device)
    probability_matrix = probability_matrix * attention_mask
    probability_matrix[:, 0] = 0  # Never mask BOS
    
    masked_indices = torch.bernoulli(probability_matrix).bool()
    input_ids[masked_indices] = mask_token_id
    labels[~masked_indices] = -100
    
    return input_ids, labels, masked_indices


# =============================================================================
# 3. VECTORIZED ADJACENT PAIR FINDING
#    Old version: nested Python for-loops over every token — O(B * L * 2R)
#    New version: fully vectorized with unfold — no Python loops
# =============================================================================
def find_adjacent_pairs_vectorized(input_ids, mask_token_id=126336, range_r=5):
    device = input_ids.device
    bsz, seq_len = input_ids.shape
    
    is_mask = (input_ids == mask_token_id)
    is_unmasked = ~is_mask
    
    # Get all masked positions
    mask_b, mask_pos = torch.where(is_mask)
    
    # For each masked position, check offsets -range_r to +range_r (excluding 0)
    offsets = torch.arange(-range_r, range_r + 1, device=device)
    offsets = offsets[offsets != 0]  # remove 0
    
    # Expand: [N_masked, num_offsets]
    anchor_candidates = mask_pos.unsqueeze(1) + offsets.unsqueeze(0)
    batch_expanded = mask_b.unsqueeze(1).expand_as(anchor_candidates)
    
    # Bounds check
    valid_bounds = (anchor_candidates >= 0) & (anchor_candidates < seq_len)
    
    # Check if anchor is unmasked
    anchor_candidates_clamped = anchor_candidates.clamp(0, seq_len - 1)
    is_anchor_unmasked = is_unmasked[batch_expanded, anchor_candidates_clamped]
    
    valid = valid_bounds & is_anchor_unmasked
    
    # Gather valid pairs
    batch_idx = batch_expanded[valid]
    anchor_pos = anchor_candidates[valid]
    masked_pos = mask_pos.unsqueeze(1).expand_as(anchor_candidates)[valid]
    
    return batch_idx, anchor_pos, masked_pos

# =============================================================================
# 4. VALIDATION
# =============================================================================
@torch.no_grad()
def evaluate(router, base_llada, loader, device, mask_token_id, alpha_base=0.05, alpha_scale=0.25):
    """
    Validation loss with adaptive alpha matching training schedule.
    Uses fixed p_mask=0.7 to evaluate in the high-mask regime we care about.
    """
    router.eval()
    total_loss = 0.0
    num_batches = 0
    p_mask_eval = 0.7  # Evaluate in the regime that matters
    alpha = alpha_base + alpha_scale * p_mask_eval  # ~0.225
    
    for i, batch in enumerate(loader):
        if i >= 50:
            break
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)

        masked_ids, labels, mask_indices = apply_random_mask(
            input_ids, attention_mask, p_mask_eval, mask_token_id
        )

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = base_llada(masked_ids, output_hidden_states=True)
        h_L = outputs.hidden_states[-1]
        
        # Vectorized pair finding
        batch_idx, anchor_pos, masked_pos = find_adjacent_pairs_vectorized(
            masked_ids, mask_token_id, range_r=5
        )
        
        if len(batch_idx) == 0:
            continue
        
        # Gather hidden states for all pairs at once
        h_anchor = h_L[batch_idx, anchor_pos]   # [N_pairs, d_model]
        h_mask = h_L[batch_idx, masked_pos]      # [N_pairs, d_model]
        target_labels = labels[batch_idx, masked_pos]  # [N_pairs]
        
        # Filter out pairs where target is -100 (not a real masked target)
        valid = target_labels != -100
        if valid.sum() == 0:
            continue
        h_anchor = h_anchor[valid]
        h_mask = h_mask[valid]
        target_labels = target_labels[valid]
        
        delta = router(h_anchor, h_mask)
        h_blended = (1 - alpha) * h_mask + alpha * delta
        
        logits = base_llada.model.transformer.ff_out(
            h_blended.to(base_llada.model.transformer.ff_out.weight.dtype)
        )
        if base_llada.model.config.scale_logits:
            logits = logits * (1 / (base_llada.model.config.d_model ** 0.5))
        
        loss = F.cross_entropy(logits, target_labels)
        total_loss += loss.item()
        num_batches += 1
    
    router.train()
    return total_loss / num_batches if num_batches > 0 else float('inf')


# =============================================================================
# 5. MAIN TRAINING LOOP
# =============================================================================
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "GSAI-ML/LLaDA-8B-Instruct"
    mask_token_id = 126336
    
    # ------------------------------------------------------------------
    # Model & tokenizer
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    base_llada = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, device_map="auto",
        torch_dtype=torch.bfloat16,  # Load natively in bf16 — halves VRAM, avoids conversion overhead
    )
    base_llada.eval()  # Frozen — never trains
    for param in base_llada.parameters():
        param.requires_grad_(False)  # Save memory — no grad metadata tracked
    
    # ------------------------------------------------------------------
    # Router
    # ------------------------------------------------------------------
    router = AMIPRouter(d_model=4096, K=8).to(device).to(torch.bfloat16)
    router = torch.compile(router, mode="max-autotune")  # CUDA graphs + triton autotuning for H100
    optimizer = torch.optim.AdamW(router.parameters(), lr=5e-4, weight_decay=0.01, fused=True)  # Single-kernel optimizer step
    
    # ------------------------------------------------------------------
    # Dataset: wikitext-103 instead of wikitext-2 (~50x more data)
    # Same format, just more diverse text. Training for same number of
    # steps costs the same — we just see more unique examples.
    # ------------------------------------------------------------------
    print("Loading dataset...")

    full_dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    print("Dataset loaded.")

    print("Filtering train...")

    train_data = full_dataset["train"].filter(lambda x: len(x["text"]) > 50)
    print(f"Train filtered: {len(train_data)} examples")

    print("Filtering val...")

    val_data = full_dataset["validation"].filter(lambda x: len(x["text"]) > 50)
    print(f"Val filtered: {len(val_data)} examples")

    def collate_fn(batch):
        return tokenizer(
            [x["text"] for x in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
    print("Creating dataloaders...")

    train_loader = DataLoader(
        train_data, batch_size=16, shuffle=True, collate_fn=collate_fn,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_data, batch_size=16, shuffle=False, collate_fn=collate_fn,
        num_workers=2, pin_memory=True, persistent_workers=True,
    )

    # Gradient accumulation: effective batch = 16 * 4 = 64
    grad_accum_steps = 4
    
    # ------------------------------------------------------------------
    # Training config
    # ------------------------------------------------------------------
    alpha_base = 0.05    # Minimum alpha (used at low mask ratios)
    alpha_scale = 0.25   # alpha = alpha_base + alpha_scale * p_mask
                         # p=0.3 -> alpha=0.125, p=1.0 -> alpha=0.30
    
    best_val_loss = float('inf')
    log_interval = 200
    val_interval = 1000
    max_steps = 50000     # Adjust based on your time budget
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=max_steps, 
    eta_min=1e-5
)
    print("=" * 60)
    print("Training ALA Router (H100 optimized)")
    print(f"  Dataset:    wikitext-103")
    print(f"  Max steps:  {max_steps}")
    print(f"  Batch size: 16 x {grad_accum_steps} grad_accum = {16 * grad_accum_steps} effective")
    print(f"  Alpha:      {alpha_base} + {alpha_scale} * p_mask")
    print(f"  p_mask:     U[0.3, 1.0]")
    print(f"  Compile:    max-autotune | Optimizer: fused AdamW | Precision: bf16")
    print("=" * 60)
    
    running_loss = 0.0
    step_count = 0
    micro_step = 0
    start_time = time.time()

    for epoch in range(100):  # Outer loop in case we need multiple passes
        for batch in train_loader:
            if step_count == 0 and micro_step == 0:
                print("First training step starting...")
            if step_count >= max_steps:
                break
            
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True).clone()
            
            # ----------------------------------------------------------
            # FIX 1: Bias mask ratio toward high values [0.3, 1.0]
            # This gives the router much more training signal in the
            # regime where it's supposed to help (high mask ratios).
            # ----------------------------------------------------------
            p_mask = 0.3 + 0.7 * torch.rand(1).item()
            
            # ----------------------------------------------------------
            # FIX 2: Adaptive alpha — scales with mask ratio
            # Low mask ratio  (0.3) -> alpha ~ 0.125 (gentle correction)
            # High mask ratio (1.0) -> alpha ~ 0.30  (aggressive correction)
            # This teaches the router to be bolder when context is sparse.
            # ----------------------------------------------------------
            alpha = alpha_base + alpha_scale * p_mask
            
            masked_ids, labels, mask_indices = apply_random_mask(
                input_ids, attention_mask, p_mask, mask_token_id
            )
            
            # Forward through frozen base model — autocast selects H100-optimized bf16 kernels
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                h_L = base_llada(masked_ids, output_hidden_states=True).hidden_states[-1]
            
            # ----------------------------------------------------------
            # FIX 3: Vectorized pair finding (replaces Python for-loops)
            # Old code: O(B * L * 2R) Python loop iterations
            # New code: single vectorized unfold + where operation
            # ----------------------------------------------------------
            batch_idx, anchor_pos, masked_pos = find_adjacent_pairs_vectorized(
                masked_ids, mask_token_id, range_r=5
            )

            if len(batch_idx) == 0:
                print(f"SKIP: no pairs, p_mask={p_mask:.2f}, num_masked={masked_ids.eq(mask_token_id).sum()}, seq_len={masked_ids.shape[1]}")
                continue
            
            # Gather all hidden states in one indexing operation
            h_anchor = h_L[batch_idx, anchor_pos].to(torch.bfloat16)  # [N_pairs, d_model]
            h_mask = h_L[batch_idx, masked_pos].to(torch.bfloat16)    # [N_pairs, d_model]
            target_labels = labels[batch_idx, masked_pos]               # [N_pairs]
            
            # Filter pairs where target is -100 (position wasn't actually masked)
            valid = target_labels != -100
            if valid.sum() == 0:
                print("target != -100")
                continue
            h_anchor = h_anchor[valid]
            h_mask = h_mask[valid]
            target_labels = target_labels[valid]
            # After filtering valid pairs, add:
            max_pairs = 2048
            if h_anchor.shape[0] > max_pairs:
                idx = torch.randperm(h_anchor.shape[0], device=h_anchor.device)[:max_pairs]
                h_anchor = h_anchor[idx]
                h_mask = h_mask[idx]
                target_labels = target_labels[idx]
            # Router forward: produces delta correction
            delta = router(h_anchor, h_mask)
            
            # Convex interpolation with adaptive alpha
            h_blended = (1 - alpha) * h_mask + alpha * delta
            
            # Project to vocabulary through frozen output head
            logits = base_llada.model.transformer.ff_out(
                h_blended.to(base_llada.model.transformer.ff_out.weight.dtype)
            )
            if base_llada.model.config.scale_logits:
                logits = logits * (1 / (base_llada.model.config.d_model ** 0.5))
            
            # Scale loss for gradient accumulation
            loss = F.cross_entropy(logits, target_labels) / grad_accum_steps
            loss.backward()

            micro_step += 1
            running_loss += loss.item() * grad_accum_steps  # unscale for logging

            # Optimizer step every grad_accum_steps micro-batches
            if micro_step % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(router.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)  # More memory-efficient than zero_grad()
                scheduler.step()
                step_count += 1

                # ----------------------------------------------------------
                # Logging
                # ----------------------------------------------------------
                if step_count % log_interval == 0:
                    avg_loss = running_loss / (log_interval * grad_accum_steps)
                    elapsed = time.time() - start_time
                    steps_per_sec = step_count / elapsed
                    eta_minutes = (max_steps - step_count) / steps_per_sec / 60
                    print(f"  Step {step_count}/{max_steps} | "
                          f"Loss: {avg_loss:.4f} | "
                          f"p_mask: {p_mask:.2f} | "
                          f"alpha: {alpha:.3f} | "
                          f"Pairs: {valid.sum().item()} | "
                          f"Steps/s: {steps_per_sec:.2f} | "
                          f"ETA: {eta_minutes:.1f}min")
                    running_loss = 0.0

                # ----------------------------------------------------------
                # Validation & checkpointing
                # ----------------------------------------------------------
                if step_count % val_interval == 0:
                    val_loss = evaluate(
                        router, base_llada, val_loader, device, mask_token_id,
                        alpha_base, alpha_scale
                    )
                    print(f"  >>> Validation Loss: {val_loss:.4f} (best: {best_val_loss:.4f})")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(router.state_dict(), "amip_router_best.pt")
                        print(f"  >>> New best model saved!")
        
        if step_count >= max_steps:
            break
    
    # Final save
    torch.save(router.state_dict(), "amip_router_final.pt")
    elapsed = time.time() - start_time
    print(f"\nTraining complete. {step_count} steps in {elapsed/60:.1f} minutes.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved: amip_router_best.pt, amip_router_final.pt")


if __name__ == "__main__":
    train()

