from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.data.truthfulqa import TruthfulQADatasetManager
from src.steering.mlp import SteeringMLP
from src.steering.vector_bank import VectorBank
from src.steering.apply import steering_hook
from src.utils.batching import build_prompt_answer_batch
from src.utils.scoring import compute_answer_logprobs

logger = logging.getLogger(__name__)


@dataclass
class MCTrainingConfig:
    epochs: int = 1
    steps_per_epoch: int = 50
    batch_size: int = 8
    margin: float = 1.0
    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    mse_reg: float = 1e-2  # MSE regularization to keep MLP close to identity
    kl_reg: float = 0.0    # KL divergence regularization (output distribution preservation)
    gradient_accumulation_steps: int = 1  # Accumulate gradients over N steps


@dataclass
class GenTrainingConfig:
    epochs: int = 1
    steps_per_epoch: int = 40
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    mse_reg: float = 1e-2  # MSE regularization to keep MLP close to identity
    kl_reg: float = 0.0    # KL divergence regularization (output distribution preservation)
    gradient_accumulation_steps: int = 1  # Accumulate gradients over N steps


def _select_mc_answers(dataset, item: dict) -> tuple[str, str] | None:
    """Select correct and incorrect MC answers using the dataset's mc_lookup."""
    question = item.get("question", "")
    mc = dataset.get_mc_targets(question)
    if not mc:
        return None
    choices = mc.get("choices") or []
    labels = mc.get("labels") or []
    if not choices or not labels:
        return None
    correct = [choice for choice, label in zip(choices, labels) if label == 1]
    incorrect = [choice for choice, label in zip(choices, labels) if label == 0]
    if not correct or not incorrect:
        return None
    return correct[0], incorrect[0]


def train_mc_mlp(
    mlp: SteeringMLP,
    *,
    model,
    tokenizer,
    dataset: TruthfulQADatasetManager,
    train_indices: Sequence[int],
    vector_bank: VectorBank,
    layer_index: int,
    primary_device: torch.device,
    max_length: int,
    config: MCTrainingConfig,
    seed: int,
) -> Dict[str, List[float]]:
    valid_indices = [idx for idx in train_indices if dataset.is_valid_mc(idx)]
    if not valid_indices:
        logger.warning("No valid MC items available for MC MLP training; skipping")
        mlp.eval()
        return {"loss": [], "margin": [], "accuracy": []}

    rng = np.random.default_rng(seed)

    # Ensure MLP matches model dtype
    param_dtype = next(model.parameters()).dtype
    mlp = mlp.to(dtype=param_dtype)

    optimizer = torch.optim.AdamW(
        mlp.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    history_loss: List[float] = []
    history_margin: List[float] = []
    history_accuracy: List[float] = []

    mlp.train()
    model.eval()

    accum_steps = config.gradient_accumulation_steps

    for epoch in range(config.epochs):
        optimizer.zero_grad()
        accum_loss = 0.0
        accum_margin = 0.0
        accum_acc = 0.0
        accum_count = 0

        for step in range(config.steps_per_epoch):
            batch_indices = rng.choice(
                valid_indices,
                size=min(config.batch_size, len(valid_indices)),
                replace=False,
            )

            prompts: List[str] = []
            answers_correct: List[str] = []
            answers_incorrect: List[str] = []

            for idx in batch_indices:
                item = dataset.get_item(int(idx))
                qa_pair = _select_mc_answers(dataset, item)
                if qa_pair is None:
                    continue
                correct_answer, incorrect_answer = qa_pair
                question = item["question"]
                prompt = f"Question: {question}\nAnswer:"
                prompts.append(prompt)
                answers_correct.append(correct_answer)
                answers_incorrect.append(incorrect_answer)

            if not prompts:
                continue

            vector = vector_bank.sample_interpolated(rng).to(primary_device, dtype=param_dtype)
            transformed = mlp(vector.unsqueeze(0)).squeeze(0)

            # KL: get baseline logits BEFORE steering (no grad, no extra memory)
            need_kl = config.kl_reg > 0
            baseline_probs = None
            if need_kl:
                inputs_correct = build_prompt_answer_batch(
                    tokenizer, prompts, answers_correct, max_length=max_length
                )
                input_ids_c, attn_c, mask_c = inputs_correct
                input_ids_c = input_ids_c.to(primary_device)
                attn_c = attn_c.to(primary_device)
                mask_c = mask_c.to(primary_device)

                with torch.no_grad():
                    baseline_out = model(input_ids=input_ids_c, attention_mask=attn_c)
                    prompt_lens = mask_c.shape[1] - mask_c.sum(dim=1).long()
                    last_pos = (prompt_lens - 1).clamp(min=0)
                    batch_idx = torch.arange(input_ids_c.shape[0], device=primary_device)
                    baseline_probs = F.softmax(baseline_out.logits[batch_idx, last_pos], dim=-1)
                    del baseline_out
                torch.cuda.empty_cache()

            with steering_hook(
                model,
                layer_index,
                transformed,
                scale=1.0,
            ):
                if not need_kl:
                    inputs_correct = build_prompt_answer_batch(
                        tokenizer, prompts, answers_correct, max_length=max_length
                    )
                    input_ids_c, attn_c, mask_c = inputs_correct
                    input_ids_c = input_ids_c.to(primary_device)
                    attn_c = attn_c.to(primary_device)
                    mask_c = mask_c.to(primary_device)

                inputs_incorrect = build_prompt_answer_batch(
                    tokenizer, prompts, answers_incorrect, max_length=max_length
                )
                input_ids_i, attn_i, mask_i = inputs_incorrect
                input_ids_i = input_ids_i.to(primary_device)
                attn_i = attn_i.to(primary_device)
                mask_i = mask_i.to(primary_device)

                # Get steered logprobs + optionally last-prompt logits for KL
                result_c = compute_answer_logprobs(
                    model,
                    input_ids=input_ids_c,
                    attention_mask=attn_c,
                    answer_mask=mask_c,
                    return_last_prompt_logits=need_kl,
                )
                if need_kl:
                    logprob_correct, _, steered_logits_last = result_c
                else:
                    logprob_correct, _ = result_c

                logprob_incorrect, _ = compute_answer_logprobs(
                    model,
                    input_ids=input_ids_i,
                    attention_mask=attn_i,
                    answer_mask=mask_i,
                )

            margin_values = logprob_incorrect - logprob_correct + config.margin
            hinge_loss = F.relu(margin_values).mean()

            # MSE regularization: encourage MLP to stay close to identity
            mse_loss = F.mse_loss(transformed, vector)

            # KL divergence: reuses logits from the existing steered forward pass
            # No extra forward pass needed — just one baseline pass (no grad)
            kl_loss = torch.tensor(0.0, device=primary_device)
            if need_kl:
                steered_log_probs = F.log_softmax(steered_logits_last, dim=-1)
                kl_loss = F.kl_div(steered_log_probs, baseline_probs, reduction='batchmean')

            loss = hinge_loss + config.mse_reg * mse_loss + config.kl_reg * kl_loss

            # Scale loss for gradient accumulation
            scaled_loss = loss / accum_steps
            scaled_loss.backward()

            # Track metrics
            with torch.no_grad():
                acc = (logprob_correct > logprob_incorrect).float().mean().item()
                accum_loss += loss.item()
                accum_margin += margin_values.mean().item()
                accum_acc += acc
                accum_count += 1

            # Update weights every accum_steps or at end of epoch
            if (step + 1) % accum_steps == 0 or (step + 1) == config.steps_per_epoch:
                if config.grad_clip:
                    torch.nn.utils.clip_grad_norm_(mlp.parameters(), config.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

                if accum_count > 0:
                    history_loss.append(accum_loss / accum_count)
                    history_margin.append(accum_margin / accum_count)
                    history_accuracy.append(accum_acc / accum_count)
                accum_loss = 0.0
                accum_margin = 0.0
                accum_acc = 0.0
                accum_count = 0

        if history_loss:
            logger.info(
                "MC MLP epoch %d/%d - loss %.4f, margin %.4f, acc %.3f",
                epoch + 1,
                config.epochs,
                history_loss[-1],
                history_margin[-1],
                history_accuracy[-1],
            )
        else:
            logger.warning("MC MLP epoch %d/%d - no valid training batches", epoch + 1, config.epochs)

    mlp.eval()
    return {
        "loss": history_loss,
        "margin": history_margin,
        "accuracy": history_accuracy,
    }


def train_gen_mlp(
    mlp: SteeringMLP,
    *,
    model,
    tokenizer,
    dataset: TruthfulQADatasetManager,
    train_indices: Sequence[int],
    vector_bank: VectorBank,
    layer_index: int,
    primary_device: torch.device,
    max_length: int,
    config: GenTrainingConfig,
    seed: int,
) -> Dict[str, List[float]]:
    rng = np.random.default_rng(seed)

    # Ensure MLP matches model dtype
    param_dtype = next(model.parameters()).dtype
    mlp = mlp.to(dtype=param_dtype)

    optimizer = torch.optim.AdamW(
        mlp.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    history_loss: List[float] = []

    mlp.train()
    model.eval()

    accum_steps = config.gradient_accumulation_steps

    for epoch in range(config.epochs):
        optimizer.zero_grad()
        accum_loss = 0.0
        accum_count = 0

        for step in range(config.steps_per_epoch):
            batch_indices = rng.choice(
                train_indices,
                size=min(config.batch_size, len(train_indices)),
                replace=False,
            )

            prompts: List[str] = []
            answers: List[str] = []

            for idx in batch_indices:
                item = dataset.get_item(int(idx))
                question = item["question"]
                correct_answers = item.get("correct_answers") or []
                best_answer = item.get("best_answer") or (correct_answers[0] if correct_answers else None)
                if not best_answer:
                    continue
                prompt = f"Question: {question}\nAnswer:"
                prompts.append(prompt)
                answers.append(best_answer)

            if not prompts:
                continue

            vector = vector_bank.sample_interpolated(rng).to(primary_device, dtype=param_dtype)
            transformed = mlp(vector.unsqueeze(0)).squeeze(0)

            need_kl = config.kl_reg > 0
            baseline_probs = None

            # Build batch once, reuse for baseline + steered
            inputs = build_prompt_answer_batch(
                tokenizer, prompts, answers, max_length=max_length
            )
            input_ids, attention_mask, answer_mask = inputs
            input_ids = input_ids.to(primary_device)
            attention_mask = attention_mask.to(primary_device)
            answer_mask = answer_mask.to(primary_device)

            # KL: baseline forward before steering (no grad)
            if need_kl:
                with torch.no_grad():
                    baseline_out = model(input_ids=input_ids, attention_mask=attention_mask)
                    prompt_lens = answer_mask.shape[1] - answer_mask.sum(dim=1).long()
                    last_pos = (prompt_lens - 1).clamp(min=0)
                    batch_idx = torch.arange(input_ids.shape[0], device=primary_device)
                    baseline_probs = F.softmax(baseline_out.logits[batch_idx, last_pos], dim=-1)
                    del baseline_out
                torch.cuda.empty_cache()

            with steering_hook(
                model,
                layer_index,
                transformed,
                scale=1.0,
            ):
                result = compute_answer_logprobs(
                    model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    answer_mask=answer_mask,
                    return_last_prompt_logits=need_kl,
                )
                if need_kl:
                    avg_logprob, _, steered_logits_last = result
                else:
                    avg_logprob, _ = result

            nll_loss = -avg_logprob.mean()

            # MSE regularization: encourage MLP to stay close to identity
            mse_loss = F.mse_loss(transformed, vector)

            # KL divergence: reuses logits from existing forward pass
            kl_loss = torch.tensor(0.0, device=primary_device)
            if need_kl:
                steered_log_probs = F.log_softmax(steered_logits_last, dim=-1)
                kl_loss = F.kl_div(steered_log_probs, baseline_probs, reduction='batchmean')

            loss = nll_loss + config.mse_reg * mse_loss + config.kl_reg * kl_loss

            # Scale loss for gradient accumulation
            scaled_loss = loss / accum_steps
            scaled_loss.backward()

            # Track metrics
            accum_loss += loss.item()
            accum_count += 1

            # Update weights every accum_steps or at end of epoch
            if (step + 1) % accum_steps == 0 or (step + 1) == config.steps_per_epoch:
                if config.grad_clip:
                    torch.nn.utils.clip_grad_norm_(mlp.parameters(), config.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

                if accum_count > 0:
                    history_loss.append(accum_loss / accum_count)
                accum_loss = 0.0
                accum_count = 0

        if history_loss:
            logger.info(
                "Gen MLP epoch %d/%d - loss %.4f",
                epoch + 1,
                config.epochs,
                history_loss[-1],
            )
        else:
            logger.warning("Gen MLP epoch %d/%d - no valid training batches", epoch + 1, config.epochs)

    mlp.eval()
    return {"loss": history_loss}
