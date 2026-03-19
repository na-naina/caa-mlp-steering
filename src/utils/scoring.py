from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def compute_answer_logprobs(
    model,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    answer_mask: torch.Tensor,
    return_last_prompt_logits: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return average log probability for answer tokens per sample.

    If return_last_prompt_logits=True, also returns the raw logits at the
    last prompt position (for KL divergence regularization).
    """

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    target_ids = input_ids[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)
    gathered = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

    masked = gathered * answer_mask
    token_counts = answer_mask.sum(dim=1)
    safe_counts = torch.clamp(token_counts, min=1.0)
    avg_logprob = masked.sum(dim=1) / safe_counts

    if return_last_prompt_logits:
        # Last prompt position = first answer token - 1 in the shifted logits
        prompt_lens = answer_mask.shape[1] - answer_mask.sum(dim=1).long()
        last_pos = (prompt_lens - 1).clamp(min=0)
        batch_idx = torch.arange(logits.shape[0], device=logits.device)
        last_prompt_logits = outputs.logits[batch_idx, last_pos]  # [batch, vocab]
        return avg_logprob, token_counts, last_prompt_logits

    return avg_logprob, token_counts
