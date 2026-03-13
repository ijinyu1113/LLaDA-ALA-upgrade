import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import random
import functools
from models import AMIPRouterTrain, ALPHA_BASE, ALPHA_SCALE, MASK_TOKEN_ID, RANGE_R
# functools.partial wraps print so flush=True is always passed
# This forces output to appear immediately in SLURM logs (otherwise Python buffers it)
print = functools.partial(print, flush=True)

# H100/GH200 optimizations — enable TF32 math for ~3x speedup with negligible precision loss
torch.set_float32_matmul_precision('high')  # allow TF32 for matmuls
torch.backends.cudnn.benchmark = True       # auto-tune convolution kernels for fixed input sizes
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
    labels = input_ids.clone()  # save original tokens — these become the ground-truth targets

    # Create probability matrix: each position gets probability p_mask of being masked
    probability_matrix = torch.full(labels.shape, p_mask, device=input_ids.device)  # [B, L] filled with p_mask
    probability_matrix = probability_matrix * attention_mask  # zero out padding (attention_mask=0 for pad tokens)
    probability_matrix[:, 0] = 0  # never mask BOS (beginning of sequence) token

    # torch.bernoulli: for each position, flip a biased coin with the given probability
    # Returns 1.0 with probability p_mask, 0.0 otherwise; .bool() converts to True/False
    masked_indices = torch.bernoulli(probability_matrix).bool()
    input_ids[masked_indices] = mask_token_id       # replace selected positions with [MASK] token (126336)
    labels[~masked_indices] = -100                   # -100 = PyTorch convention for "ignore in loss computation"

    return input_ids, labels, masked_indices


# =============================================================================
# 3. VECTORIZED ADJACENT PAIR FINDING
# =============================================================================
def find_adjacent_pairs_vectorized(input_ids, mask_token_id=MASK_TOKEN_ID, range_r=RANGE_R):
    """
    For every masked token, find all unmasked tokens within range_r positions.
    Returns flat arrays of (batch_index, anchor_position, masked_position) triplets.

    Example: if token at position 5 is masked and tokens at positions 3, 7, 8 are unmasked
    and range_r=10, this produces 3 pairs: (5,3), (5,7), (5,8).
    """
    device = input_ids.device
    bsz, seq_len = input_ids.shape

    is_mask = (input_ids == mask_token_id)     # [B, L] — True where token is [MASK]
    is_unmasked = ~is_mask                      # [B, L] — True where token is NOT [MASK]

    # torch.where on a 2D tensor returns (row_indices, col_indices) where condition is True
    # mask_b = which batch element each masked token belongs to
    # mask_pos = which sequence position each masked token is at
    mask_b, mask_pos = torch.where(is_mask)     # each is 1D tensor of length N_masked

    # Offsets: [-range_r, ..., -1, 1, ..., range_r] — positions to check around each mask
    offsets = torch.arange(-range_r, range_r + 1, device=device)
    offsets = offsets[offsets != 0]  # remove 0 (don't pair a token with itself) -> 2*range_r offsets

    # --- Broadcasting trick to create all candidate pairs at once ---
    # mask_pos is [N_masked], offsets is [2*range_r]
    # unsqueeze(1) makes mask_pos into [N_masked, 1]
    # unsqueeze(0) makes offsets into [1, 2*range_r]
    # Adding them broadcasts: [N_masked, 1] + [1, 2*range_r] -> [N_masked, 2*range_r]
    # Each row = one masked token, each column = one candidate neighbor position
    anchor_candidates = mask_pos.unsqueeze(1) + offsets.unsqueeze(0)  # [N_masked, 2*range_r]

    # Expand batch indices to match: same batch index for all candidates of each masked token
    batch_expanded = mask_b.unsqueeze(1).expand_as(anchor_candidates)  # [N_masked, 2*range_r]

    # --- Filter invalid candidates ---
    # Filter 1: position must be within sequence bounds [0, seq_len)
    valid_bounds = (anchor_candidates >= 0) & (anchor_candidates < seq_len)

    # Filter 2: candidate position must be an unmasked token (not another [MASK])
    # Clamp first to avoid out-of-bounds indexing error, then check the boolean mask
    anchor_candidates_clamped = anchor_candidates.clamp(0, seq_len - 1)
    # Advanced 2D indexing: is_unmasked[batch_idx, position] for every candidate
    is_anchor_unmasked = is_unmasked[batch_expanded, anchor_candidates_clamped]

    # Combine both filters
    valid = valid_bounds & is_anchor_unmasked    # [N_masked, 2*range_r] — True where valid pair exists

    # --- Extract valid pairs using boolean indexing (flattens the 2D grid to 1D) ---
    batch_idx = batch_expanded[valid]            # [N_valid_pairs] — which batch element
    anchor_pos = anchor_candidates[valid]        # [N_valid_pairs] — position of unmasked anchor
    # Need to also know which masked position each pair came from
    masked_pos = mask_pos.unsqueeze(1).expand_as(anchor_candidates)[valid]  # [N_valid_pairs]

    return batch_idx, anchor_pos, masked_pos

# =============================================================================
# 4. VALIDATION
# =============================================================================
@torch.no_grad()  # disable gradient computation entirely — faster + less GPU memory
def evaluate(router, base_llada, loader, device, mask_token_id=MASK_TOKEN_ID, alpha_base=ALPHA_BASE, alpha_scale=ALPHA_SCALE):
    """
    Validation loss with adaptive alpha matching training schedule.
    Uses fixed p_mask=0.7 to evaluate in the high-mask regime we care about.
    """
    router.eval()  # set to eval mode (affects dropout/batchnorm — good practice even if not used)
    total_loss = 0.0
    num_batches = 0
    p_mask_eval = 0.7
    alpha = alpha_base + alpha_scale * p_mask_eval

    for i, batch in enumerate(loader):
        if i >= 30:  # only evaluate on 30 batches for speed
            break
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)

        masked_ids, labels, mask_indices = apply_random_mask(
            input_ids, attention_mask, p_mask_eval, mask_token_id
        )

        # Run frozen base model in mixed precision (bfloat16 = half precision, 2x faster)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = base_llada(masked_ids, output_hidden_states=True)
        h_L = outputs.hidden_states[-1]  # last layer hidden states [B, L, d]

        batch_idx, anchor_pos, masked_pos = find_adjacent_pairs_vectorized(
            masked_ids, mask_token_id, range_r=RANGE_R
        )

        if len(batch_idx) == 0:
            continue

        # Advanced indexing: h_L[batch_idx, anchor_pos] picks one hidden vector per pair
        # batch_idx selects which sample in the batch, anchor_pos selects which position
        # Result shape: [N_pairs, d_model]
        h_anchor = h_L[batch_idx, anchor_pos]
        h_mask = h_L[batch_idx, masked_pos]
        target_labels = labels[batch_idx, masked_pos]  # ground-truth token at each masked position

        # Filter out pairs where target is -100 (padding or non-target positions)
        valid = target_labels != -100
        if valid.sum() == 0:
            continue
        h_anchor = h_anchor[valid]       # boolean indexing: keep only rows where valid=True
        h_mask = h_mask[valid]
        target_labels = target_labels[valid]

        delta = router(h_anchor, h_mask)
        h_blended = h_mask + alpha * delta  # residual correction

        # Project to vocabulary logits using frozen output head (ff_out)
        # Cast to match ff_out's weight dtype (might differ from bfloat16)
        logits = base_llada.model.transformer.ff_out(
            h_blended.to(base_llada.model.transformer.ff_out.weight.dtype)
        )
        if base_llada.model.config.scale_logits:
            logits = logits * (1 / (base_llada.model.config.d_model ** 0.5))

        # Cross-entropy: standard classification loss — how well does the model predict the correct token?
        loss = F.cross_entropy(logits, target_labels)
        total_loss += loss.item()  # .item() converts single-element tensor to Python float
        num_batches += 1

    router.train()  # set back to training mode
    return total_loss / num_batches if num_batches > 0 else float('inf')


# =============================================================================
# 5. MAIN TRAINING LOOP
# =============================================================================
def parse_train_args():
    parser = argparse.ArgumentParser(description="Train ALA router (staged)")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2],
                        help="Stage 1: math-heavy from scratch, Stage 2: gate-only on balanced data")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to load router checkpoint from (required for stage 2)")
    parser.add_argument("--save-prefix", type=str, default="amip_router",
                        help="Prefix for saved checkpoints (e.g. amip_router_math)")
    return parser.parse_args()


def train():
    args = parse_train_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "GSAI-ML/LLaDA-8B-Instruct"
    mask_token_id = MASK_TOKEN_ID

    # ------------------------------------------------------------------
    # Stage-dependent config
    # ------------------------------------------------------------------
    if args.stage == 1:
        use_gate = False
        max_steps = 10000
        lr = 5e-4
        use_diverse = False  # math-heavy: Wiki 90K + GSM8K 12x + MATH 12x
    elif args.stage == 2:
        use_gate = True
        max_steps = 5000
        lr = 5e-4           # only gate params, can keep LR high
        use_diverse = True   # balanced: Wiki 90K + Math 6x + Diverse 3x

    print(f"Stage {args.stage}: use_gate={use_gate}, max_steps={max_steps}, lr={lr}, diverse={use_diverse}")

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
    router = AMIPRouterTrain(d_model=4096, K=8, use_gate=use_gate).to(device).to(torch.bfloat16)

    # Load checkpoint (required for stage 2)
    if args.checkpoint:
        state_dict = torch.load(args.checkpoint, map_location=device)
        # Strip _orig_mod. prefix if saved from torch.compile
        cleaned = {}
        for k, v in state_dict.items():
            cleaned[k.replace("_orig_mod.", "")] = v
        missing, unexpected = router.load_state_dict(cleaned, strict=False)
        print(f"  Loaded checkpoint: {args.checkpoint}")
        if missing:
            print(f"  Missing keys (newly added): {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")

    # Stage 2: freeze everything except conf_gate
    if args.stage == 2:
        for name, param in router.named_parameters():
            if "conf_gate" not in name:
                param.requires_grad_(False)
        trainable = sum(p.numel() for p in router.parameters() if p.requires_grad)
        total = sum(p.numel() for p in router.parameters())
        print(f"  Stage 2: {trainable} trainable / {total} total params")

    # Compile AFTER checkpoint load + freeze
    router = torch.compile(router, mode="max-autotune")
    optimizer = torch.optim.AdamW(
        [p for p in router.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01, fused=True
    )

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------
    print("Loading datasets...")

    # --- Wiki (always included, downsample to ~90K) ---
    wiki = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    wiki_train = wiki["train"].filter(lambda x: len(x["text"]) > 50)
    wiki_val = wiki["validation"].filter(lambda x: len(x["text"]) > 50)
    wiki_train = wiki_train.select(range(min(90000, len(wiki_train))))
    print(f"  Wiki train: {len(wiki_train)}")

    # --- Math ---
    gsm8k = load_dataset("openai/gsm8k", "main", split="train")
    gsm8k = gsm8k.map(lambda x: {"text": f"Question: {x['question']}\nAnswer: {x['answer']}"})
    gsm8k = gsm8k.remove_columns([c for c in gsm8k.column_names if c != "text"])

    math_subjects = ["algebra", "counting_and_probability", "geometry",
                     "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
    math_parts = [load_dataset("EleutherAI/hendrycks_math", subj, split="train") for subj in math_subjects]
    math_ds = concatenate_datasets(math_parts)
    math_ds = math_ds.map(lambda x: {"text": f"Problem: {x['problem']}\nSolution: {x['solution']}"})
    math_ds = math_ds.remove_columns([c for c in math_ds.column_names if c != "text"])

    math_combined = concatenate_datasets([gsm8k, math_ds])

    if use_diverse:
        # Stage 2: balanced mix — Math 6x (~90K)
        math_repeat = 6
    else:
        # Stage 1: math-heavy — GSM8K 12x + MATH 12x (separate upsampling)
        math_repeat = 12

    if not use_diverse:
        # Stage 1: upsample GSM8K and MATH separately at 12x each
        gsm8k_up = concatenate_datasets([gsm8k] * math_repeat)
        math_ds_up = concatenate_datasets([math_ds] * math_repeat)
        math_up = concatenate_datasets([gsm8k_up, math_ds_up])
        print(f"  Math: GSM8K {len(gsm8k)} x {math_repeat} + MATH {len(math_ds)} x {math_repeat} = {len(math_up)}")
    else:
        # Stage 2: combined 6x
        math_up = concatenate_datasets([math_combined] * math_repeat)
        print(f"  Math: GSM8K {len(gsm8k)} + MATH {len(math_ds)} = {len(math_combined)} x {math_repeat} = {len(math_up)}")

    if use_diverse:
        # --- Diverse reasoning (SciQ + ECQA + CoS-E + OpenBookQA, ~3x upsample → ~90K) ---
        sciq = load_dataset("allenai/sciq", split="train")
        def _format_sciq(x, idx):
            labels = ["A", "B", "C", "D"]
            choices = [(x["correct_answer"], True), (x["distractor1"], False),
                       (x["distractor2"], False), (x["distractor3"], False)]
            rng = random.Random(idx)
            rng.shuffle(choices)
            gold_label = next(labels[i] for i, (_, c) in enumerate(choices) if c)
            choice_str = " ".join(f"({labels[i]}) {t}" for i, (t, _) in enumerate(choices))
            return {"text": (
                f"Question: {x['question']}\n{choice_str}\n"
                f"Let's think step by step. {x['support']}\n"
                f"Therefore the answer is ({gold_label})."
            )}
        sciq = sciq.map(_format_sciq, with_indices=True)
        sciq = sciq.remove_columns([c for c in sciq.column_names if c != "text"])

        ecqa = load_dataset("tasksource/ecqa", split="train")
        ecqa = ecqa.map(lambda x: {"text": (
            f"Question: {x['q_text']}\n"
            f"(1) {x['q_op1']} (2) {x['q_op2']} (3) {x['q_op3']} "
            f"(4) {x['q_op4']} (5) {x['q_op5']}\n"
            f"Let's think step by step. {x['taskB']}\n"
            f"Therefore the answer is {x['q_ans']}."
        )})
        ecqa = ecqa.remove_columns([c for c in ecqa.column_names if c != "text"])

        cose = load_dataset("Salesforce/cos_e", "v1.11", split="train")
        cose = cose.map(lambda x: {"text": (
            f"Question: {x['question']}\n"
            f"Choices: {', '.join(x['choices'])}\n"
            f"Let's think step by step. {x['abstractive_explanation']}\n"
            f"Therefore the answer is {x['answer']}."
        )})
        cose = cose.remove_columns([c for c in cose.column_names if c != "text"])

        obqa = load_dataset("allenai/openbookqa", "additional", split="train")
        obqa = obqa.map(lambda x: {"text": (
            f"Question: {x['question_stem']}\n"
            + "\n".join(f"{l}) {t}" for l, t in zip(x['choices']['label'], x['choices']['text']))
            + f"\nLet's think step by step. {x['fact1']}\n"
            f"Therefore the answer is {x['answerKey']}."
        )})
        obqa = obqa.remove_columns([c for c in obqa.column_names if c != "text"])

        diverse = concatenate_datasets([sciq, ecqa, cose, obqa])
        diverse_repeat = 3
        diverse_up = concatenate_datasets([diverse] * diverse_repeat)
        print(f"  Diverse: SciQ {len(sciq)} + ECQA {len(ecqa)} + CoS-E {len(cose)} + OBQA {len(obqa)} "
              f"= {len(diverse)} x {diverse_repeat} = {len(diverse_up)}")
        train_data = concatenate_datasets([wiki_train, math_up, diverse_up])
    else:
        train_data = concatenate_datasets([wiki_train, math_up])

    val_data = wiki_val
    print(f"  Total train: {len(train_data)}")

    def collate_fn(batch):
        """Tokenize a batch of text strings into padded tensors."""
        return tokenizer(
            [x["text"] for x in batch],
            return_tensors="pt",  # return PyTorch tensors (not lists)
            padding=True,         # pad shorter sequences to match longest in batch
            truncation=True,      # truncate sequences longer than max_length
            max_length=256
        )
    print("Creating dataloaders...")

    train_loader = DataLoader(
        train_data, batch_size=8, shuffle=True, collate_fn=collate_fn,
        num_workers=4,           # 4 CPU processes to load/tokenize data in parallel
        pin_memory=True,         # pin data in CPU memory for faster GPU transfer
        persistent_workers=True, # keep worker processes alive between epochs (avoid respawn cost)
    )
    val_loader = DataLoader(
        val_data, batch_size=8, shuffle=False, collate_fn=collate_fn,
        num_workers=2, pin_memory=True, persistent_workers=True,
    )

    # Gradient accumulation: process 4 micro-batches before updating weights
    # This simulates a larger effective batch size (8 * 4 = 32) without needing 4x GPU memory
    grad_accum_steps = 4

    # ------------------------------------------------------------------
    # Training config
    # ------------------------------------------------------------------
    alpha_base = 0.1   # train with larger alpha so router learns stronger corrections
    alpha_scale = ALPHA_SCALE

    best_val_loss = float('inf')
    log_interval = 100
    val_interval = 500

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_steps, eta_min=1e-5
    )
    save_best = f"{args.save_prefix}_best.pt"
    save_final = f"{args.save_prefix}_final.pt"

    print("=" * 60)
    print(f"Training ALA Router — Stage {args.stage}")
    print(f"  Save prefix: {args.save_prefix}")
    print(f"  Max steps:  {max_steps}")
    print(f"  LR:         {lr}")
    print(f"  Batch size: 8 x {grad_accum_steps} grad_accum = {8 * grad_accum_steps} effective")
    print(f"  Alpha:      {alpha_base} + {alpha_scale} * p_mask")
    print(f"  Gate:       {use_gate}")
    print(f"  range_r:    {RANGE_R}")
    print(f"  p_mask:     U[0.3, 1.0]")
    print("=" * 60)

    running_loss = 0.0
    step_count = 0      # optimizer steps (each = grad_accum_steps micro-batches)
    micro_step = 0       # individual forward/backward passes
    start_time = time.time()

    for epoch in range(100):  # outer loop — will break early at max_steps
        for batch in train_loader:
            if step_count == 0 and micro_step == 0:
                print("First training step starting...")
            if step_count >= max_steps:
                break

            # Transfer data to GPU asynchronously (non_blocking=True lets CPU continue while data transfers)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True).clone()  # clone: don't modify original

            # Sample mask ratio uniformly from [0.3, 1.0]
            # 0.3 + 0.7 * U[0,1] = U[0.3, 1.0] — train on varying mask ratios
            p_mask = 0.3 + 0.7 * torch.rand(1).item()  # .item() converts tensor to Python float
            alpha = alpha_base + alpha_scale * p_mask    # with ALPHA_SCALE=0, always 0.1

            masked_ids, labels, mask_indices = apply_random_mask(
                input_ids, attention_mask, p_mask, mask_token_id
            )

            # --- Base model forward (frozen, no gradients) ---
            # torch.no_grad(): don't track operations for gradient computation (saves memory)
            # torch.amp.autocast: automatically use bfloat16 for matrix ops (2x faster on GPU)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                h_L = base_llada(masked_ids, output_hidden_states=True).hidden_states[-1]

            batch_idx, anchor_pos, masked_pos = find_adjacent_pairs_vectorized(
                masked_ids, mask_token_id, range_r=RANGE_R
            )

            if len(batch_idx) == 0:
                print(f"SKIP: no pairs, p_mask={p_mask:.2f}, num_masked={masked_ids.eq(mask_token_id).sum()}, seq_len={masked_ids.shape[1]}")
                continue

            # Extract hidden states for each (anchor, mask) pair using advanced indexing
            # h_L is [B, L, d]; indexing with [N_pairs] tensors picks out N_pairs vectors of dim d
            h_anchor = h_L[batch_idx, anchor_pos].to(torch.bfloat16)  # [N_pairs, d]
            h_mask = h_L[batch_idx, masked_pos].to(torch.bfloat16)    # [N_pairs, d]
            target_labels = labels[batch_idx, masked_pos]              # [N_pairs] — ground-truth token IDs

            # Filter out pairs where target is -100 (padding tokens)
            valid = target_labels != -100
            if valid.sum() == 0:
                print("target != -100")
                continue
            h_anchor = h_anchor[valid]         # boolean indexing keeps only valid rows
            h_mask = h_mask[valid]
            target_labels = target_labels[valid]

            # Subsample if too many pairs (prevents GPU OOM on long sequences with high mask ratio)
            # torch.randperm: random permutation of indices; [:max_pairs] takes first max_pairs
            max_pairs = 2048
            if h_anchor.shape[0] > max_pairs:
                idx = torch.randperm(h_anchor.shape[0], device=h_anchor.device)[:max_pairs]
                h_anchor = h_anchor[idx]
                h_mask = h_mask[idx]
                target_labels = target_labels[idx]

            # --- Router forward: produces relevance-gated delta correction ---
            delta = router(h_anchor, h_mask)   # [N_pairs, d] — per-pair correction vectors

            # Residual blending: original hidden state + small learned correction
            h_blended = h_mask + alpha * delta  # alpha=0.1 keeps corrections small

            # Project blended hidden states to vocabulary logits through frozen output head
            # Cast to match ff_out weight dtype (could be different from bfloat16)
            logits = base_llada.model.transformer.ff_out(
                h_blended.to(base_llada.model.transformer.ff_out.weight.dtype)
            )
            if base_llada.model.config.scale_logits:
                logits = logits * (1 / (base_llada.model.config.d_model ** 0.5))

            # --- Gradient accumulation ---
            # Divide loss by grad_accum_steps because we accumulate gradients over 4 micro-batches
            # Without dividing, the accumulated gradient would be 4x too large
            loss = F.cross_entropy(logits, target_labels) / grad_accum_steps
            loss.backward()  # compute gradients and ADD to existing gradients (accumulate)

            micro_step += 1
            running_loss += loss.item() * grad_accum_steps  # undo /4 for logging the true loss value

            # Every grad_accum_steps micro-batches, actually update the weights
            if micro_step % grad_accum_steps == 0:
                # Clip gradient norm to 1.0: if total gradient magnitude > 1.0, scale it down
                # Prevents "exploding gradients" that can destabilize training
                torch.nn.utils.clip_grad_norm_(router.parameters(), max_norm=1.0)
                optimizer.step()                          # apply gradients to update router weights
                optimizer.zero_grad(set_to_none=True)     # clear gradients (set_to_none=True saves memory vs filling with 0)
                scheduler.step()                          # advance cosine LR schedule
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
                        # Save underlying module to avoid _orig_mod. prefix from torch.compile
                        underlying = router._orig_mod if hasattr(router, '_orig_mod') else router
                        torch.save(underlying.state_dict(), save_best)
                        print(f"  >>> New best model saved to {save_best}!")

        if step_count >= max_steps:
            break

    # Final save (regardless of validation loss)
    underlying = router._orig_mod if hasattr(router, '_orig_mod') else router
    torch.save(underlying.state_dict(), save_final)
    elapsed = time.time() - start_time
    print(f"\nTraining complete. {step_count} steps in {elapsed/60:.1f} minutes.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved: {save_best}, {save_final}")


if __name__ == "__main__":
    train()
