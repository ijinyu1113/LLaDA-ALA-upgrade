import torch
import torch.nn.functional as F
import numpy as np

# ============================================================
# 2. GENERATION UTILS
# ============================================================
def add_gumbel_noise(logits, temperature):
    if temperature == 0: return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    return logits.exp() / ((- torch.log(noise)) ** temperature)


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base, remainder = mask_num // steps, mask_num % steps
    res = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        res[i, :remainder[i]] += 1
    return res


@torch.no_grad()
def generate(model, prompt_ids, steps=128, gen_length=128, block_length=32, use_router=True, temp=0.0):

    """Full LLaDA generation with block-wise confidence-based unmasking."""
    torch.manual_seed(42)
    device, mask_id = model.device, 126336
    x = torch.full((prompt_ids.shape[0], prompt_ids.shape[1] + gen_length), mask_id, dtype=torch.long).to(device)
    x[:, :prompt_ids.shape[1]] = prompt_ids.clone()

    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    for b in range(num_blocks):
        b_start = prompt_ids.shape[1] + (b * block_length)
        b_end = b_start + block_length

        block_mask = (x[:, b_start:b_end] == mask_id)
        transfer_schedule = get_num_transfer_tokens(block_mask, steps_per_block)

        for i in range(steps_per_block):
            mask_index = (x == mask_id)
            logits = model(x).logits if use_router else model.base_logits(x)
            #logits[:, :, 126081] = -torch.inf

            logits_noise = add_gumbel_noise(logits, temperature=temp)
            x0 = torch.argmax(logits_noise, dim=-1)

            probs = F.softmax(logits, dim=-1)
            x0_p = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            x0_p[:, b_end:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_idx = torch.zeros_like(x, dtype=torch.bool)
            for j in range(confidence.shape[0]):
                _, sel_idx = torch.topk(confidence[j], k=transfer_schedule[j, i])
                transfer_idx[j, sel_idx] = True
            x[transfer_idx] = x0[transfer_idx]

    return x
