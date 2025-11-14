from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch


def build_prompt_answer_batch(
    tokenizer,
    prompts: Sequence[str],
    answers: Sequence[str],
    *,
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize prompt/answer pairs and return tensors for model scoring.

    Returns a tuple ``(input_ids, attention_mask, answer_mask)`` where
    ``answer_mask`` is aligned with ``target_ids`` (i.e. ``input_ids[:, 1:]``)
    and indicates which positions correspond to answer tokens.
    """

    if len(prompts) != len(answers):
        raise ValueError("Prompts and answers must have equal length")

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        # Fall back to EOS for padding if PAD is undefined (e.g. Gemma/LLaMA)
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer requires either pad_token_id or eos_token_id")

    bos_token_id = getattr(tokenizer, "bos_token_id", None)

    sequences: List[List[int]] = []
    answer_masks: List[List[float]] = []

    for prompt, answer in zip(prompts, answers):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)

        if bos_token_id is not None:
            prompt_ids = [bos_token_id] + prompt_ids

        combined = prompt_ids + answer_ids
        if len(combined) > max_length:
            combined = combined[:max_length]
        sequences.append(combined)

        seq_len = len(combined)
        prompt_len = min(len(prompt_ids), seq_len)
        answer_len = max(0, seq_len - prompt_len)

        mask_length = max(seq_len - 1, 0)
        mask = [0.0] * mask_length
        start = max(prompt_len - 1, 0)
        end = min(prompt_len - 1 + answer_len, mask_length)
        for idx in range(start, end):
            mask[idx] = 1.0
        answer_masks.append(mask)

    if not sequences:
        raise ValueError("Received empty prompt/answer batch")

    max_seq_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)

    input_ids = torch.full((batch_size, max_seq_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    answer_mask = torch.zeros((batch_size, max_seq_len - 1), dtype=torch.float32)

    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        input_ids[i, :seq_len] = torch.tensor(seq, dtype=torch.long)
        attention_mask[i, :seq_len] = 1

        mask = answer_masks[i]
        if mask:
            answer_mask[i, : len(mask)] = torch.tensor(mask, dtype=torch.float32)

    return input_ids, attention_mask, answer_mask
