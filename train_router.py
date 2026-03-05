import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import functools
from models import AMIPRouterTrain, ALPHA_BASE, ALPHA_SCALE, MASK_TOKEN_ID, RANGE_R
print = functools.partial(print, flush=True)

# H100 optimizations
torch.set_float32_matmul_precision('high')  # TF32 tensor cores
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# =============================================================================
# 2. VECTORIZED MASKING
# =============================================================================
def apply_random_mask(input_ids, attention_mask, p_mask, mask_token_id=MASK_TOKEN_ID):
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
# =============================================================================
def find_adjacent_pairs_vectorized(input_ids, mask_token_id=MASK_TOKEN_ID, range_r=RANGE_R):
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
def evaluate(router, base_llada, loader, device, mask_token_id=MASK_TOKEN_ID, alpha_base=ALPHA_BASE, alpha_scale=ALPHA_SCALE):
    """
    Validation loss with adaptive alpha matching training schedule.
    Uses fixed p_mask=0.7 to evaluate in the high-mask regime we care about.
    """
    router.eval()
    total_loss = 0.0
    num_batches = 0
    p_mask_eval = 0.7
    alpha = alpha_base + alpha_scale * p_mask_eval

    for i, batch in enumerate(loader):
        if i >= 30:  # Faster validation
            break
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)

        masked_ids, labels, mask_indices = apply_random_mask(
            input_ids, attention_mask, p_mask_eval, mask_token_id
        )

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = base_llada(masked_ids, output_hidden_states=True)
        h_L = outputs.hidden_states[-1]

        batch_idx, anchor_pos, masked_pos = find_adjacent_pairs_vectorized(
            masked_ids, mask_token_id, range_r=RANGE_R
        )

        if len(batch_idx) == 0:
            continue

        h_anchor = h_L[batch_idx, anchor_pos]
        h_mask = h_L[batch_idx, masked_pos]
        target_labels = labels[batch_idx, masked_pos]

        valid = target_labels != -100
        if valid.sum() == 0:
            continue
        h_anchor = h_anchor[valid]
        h_mask = h_mask[valid]
        target_labels = target_labels[valid]

        delta = router(h_anchor, h_mask)
        h_blended = h_mask + alpha * delta

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
    mask_token_id = MASK_TOKEN_ID

    # ------------------------------------------------------------------
    # Model & tokenizer
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    base_llada = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    base_llada.eval()
    for param in base_llada.parameters():
        param.requires_grad_(False)

    # ------------------------------------------------------------------
    # Router
    # ------------------------------------------------------------------
    router = AMIPRouterTrain(d_model=4096, K=8).to(device).to(torch.bfloat16)
    router = torch.compile(router, mode="max-autotune")
    optimizer = torch.optim.AdamW(router.parameters(), lr=5e-4, weight_decay=0.01, fused=True)

    # ------------------------------------------------------------------
    # Dataset: wikitext-103 + GSM8K train + MATH train
    # Mixed data teaches the router to handle both natural language
    # and mathematical reasoning patterns.
    # ------------------------------------------------------------------
    print("Loading datasets...")

    # Wikitext-103
    wiki = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    wiki_train = wiki["train"].filter(lambda x: len(x["text"]) > 50)
    wiki_val = wiki["validation"].filter(lambda x: len(x["text"]) > 50)
    print(f"  Wikitext train: {len(wiki_train)}")

    # GSM8K train split — format: question + answer as "text"
    gsm8k = load_dataset("openai/gsm8k", "main", split="train")
    gsm8k = gsm8k.map(lambda x: {"text": f"Question: {x['question']}\nAnswer: {x['answer']}"})
    gsm8k = gsm8k.remove_columns([c for c in gsm8k.column_names if c != "text"])
    print(f"  GSM8K train: {len(gsm8k)}")

    # MATH train split (load all subjects from EleutherAI mirror)
    math_subjects = ["algebra", "counting_and_probability", "geometry",
                     "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
    math_parts = [load_dataset("EleutherAI/hendrycks_math", subj, split="train") for subj in math_subjects]
    math_ds = concatenate_datasets(math_parts)
    math_ds = math_ds.map(lambda x: {"text": f"Problem: {x['problem']}\nSolution: {x['solution']}"})
    math_ds = math_ds.remove_columns([c for c in math_ds.column_names if c != "text"])
    print(f"  MATH train: {len(math_ds)}")

    # Oversample math data so it's ~30% of training
    # Wiki ~180K, GSM8K ~7.5K x 12 = ~90K, MATH ~7.5K x 12 = ~90K
    # Total ~360K, math fraction ~50% — enough exposure for reasoning
    math_repeat = 12
    gsm8k_up = concatenate_datasets([gsm8k] * math_repeat)
    math_up = concatenate_datasets([math_ds] * math_repeat)
    train_data = concatenate_datasets([wiki_train, gsm8k_up, math_up])
    val_data = wiki_val
    print(f"  GSM8K upsampled: {len(gsm8k)} x {math_repeat} = {len(gsm8k_up)}")
    print(f"  MATH upsampled:  {len(math_ds)} x {math_repeat} = {len(math_up)}")
    print(f"  Combined train:  {len(train_data)} (~50% math)")

    def collate_fn(batch):
        return tokenizer(
            [x["text"] for x in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256  # Longer for math chains
        )
    print("Creating dataloaders...")

    train_loader = DataLoader(
        train_data, batch_size=8, shuffle=True, collate_fn=collate_fn,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_data, batch_size=8, shuffle=False, collate_fn=collate_fn,
        num_workers=2, pin_memory=True, persistent_workers=True,
    )

    # Gradient accumulation: effective batch = 8 * 4 = 32
    grad_accum_steps = 4

    # ------------------------------------------------------------------
    # Training config — preliminary fast run
    # ------------------------------------------------------------------
    alpha_base = ALPHA_BASE
    alpha_scale = ALPHA_SCALE

    best_val_loss = float('inf')
    log_interval = 100
    val_interval = 500
    max_steps = 10000  # Preliminary — fast iteration

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_steps, eta_min=1e-5
    )
    print("=" * 60)
    print("Training ALA Router (preliminary)")
    print(f"  Dataset:    wikitext-103 + GSM8K + MATH")
    print(f"  Max steps:  {max_steps}")
    print(f"  Batch size: 8 x {grad_accum_steps} grad_accum = {8 * grad_accum_steps} effective")
    print(f"  Alpha:      {alpha_base} + {alpha_scale} * p_mask")
    print(f"  range_r:    {RANGE_R}")
    print(f"  p_mask:     U[0.3, 1.0]")
    print(f"  Compile:    max-autotune | Optimizer: fused AdamW | Precision: bf16")
    print("=" * 60)

    running_loss = 0.0
    step_count = 0
    micro_step = 0
    start_time = time.time()

    for epoch in range(100):
        for batch in train_loader:
            if step_count == 0 and micro_step == 0:
                print("First training step starting...")
            if step_count >= max_steps:
                break

            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True).clone()

            p_mask = 0.3 + 0.7 * torch.rand(1).item()
            alpha = alpha_base + alpha_scale * p_mask

            masked_ids, labels, mask_indices = apply_random_mask(
                input_ids, attention_mask, p_mask, mask_token_id
            )

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                h_L = base_llada(masked_ids, output_hidden_states=True).hidden_states[-1]

            batch_idx, anchor_pos, masked_pos = find_adjacent_pairs_vectorized(
                masked_ids, mask_token_id, range_r=RANGE_R
            )

            if len(batch_idx) == 0:
                print(f"SKIP: no pairs, p_mask={p_mask:.2f}, num_masked={masked_ids.eq(mask_token_id).sum()}, seq_len={masked_ids.shape[1]}")
                continue

            h_anchor = h_L[batch_idx, anchor_pos].to(torch.bfloat16)
            h_mask = h_L[batch_idx, masked_pos].to(torch.bfloat16)
            target_labels = labels[batch_idx, masked_pos]

            valid = target_labels != -100
            if valid.sum() == 0:
                print("target != -100")
                continue
            h_anchor = h_anchor[valid]
            h_mask = h_mask[valid]
            target_labels = target_labels[valid]
            max_pairs = 2048
            if h_anchor.shape[0] > max_pairs:
                idx = torch.randperm(h_anchor.shape[0], device=h_anchor.device)[:max_pairs]
                h_anchor = h_anchor[idx]
                h_mask = h_mask[idx]
                target_labels = target_labels[idx]
            # Router forward: produces relevance-gated delta
            delta = router(h_anchor, h_mask)

            # Residual correction: h_mask + alpha * delta
            h_blended = h_mask + alpha * delta

            # Project to vocabulary through frozen output head
            logits = base_llada.model.transformer.ff_out(
                h_blended.to(base_llada.model.transformer.ff_out.weight.dtype)
            )
            if base_llada.model.config.scale_logits:
                logits = logits * (1 / (base_llada.model.config.d_model ** 0.5))

            loss = F.cross_entropy(logits, target_labels) / grad_accum_steps
            loss.backward()

            micro_step += 1
            running_loss += loss.item() * grad_accum_steps

            if micro_step % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(router.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                step_count += 1

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
