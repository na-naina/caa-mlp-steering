from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from src.steering.apply import steering_hook

logger = logging.getLogger(__name__)


@dataclass
class MCStats:
    accuracy: float
    avg_correct_prob: float
    avg_incorrect_prob: float
    total: int


@dataclass
class GenerationStats:
    accuracy: Optional[float]
    informativeness_mean: Optional[float]
    semantic_mean: Optional[float]
    semantic_diff_mean: Optional[float]
    semantic_acc: Optional[float]
    bleurt_mean: Optional[float]
    bleurt_diff_mean: Optional[float]
    bleurt_acc: Optional[float]
    total: int


def evaluate_multiple_choice(
    model,
    tokenizer,
    items: Sequence[dict],
    *,
    layer_index: int,
    steering_vector: Optional[torch.Tensor],
    scale: float,
    max_length: int = 512,
    primary_device: torch.device,
    seed: int = 0,
) -> Dict:
    """Evaluate binary MC questions from TruthfulQA."""
    rng = random.Random(seed)
    results = []

    steering = steering_vector if scale != 0 else None

    with steering_hook(model, layer_index, steering, scale=scale):
        for item in items:
            question = item["question"]
            mc = item.get("mc1_targets")
            if not mc:
                continue

            choices = mc["choices"]
            labels = mc["labels"]
            correct_indices = [i for i, lab in enumerate(labels) if lab == 1]
            incorrect_indices = [i for i, lab in enumerate(labels) if lab == 0]
            if not correct_indices or not incorrect_indices:
                continue

            correct_choice = choices[correct_indices[0]]
            incorrect_choice = choices[incorrect_indices[0]]

            if rng.random() < 0.5:
                option_a, option_b = correct_choice, incorrect_choice
                correct_label = "A"
            else:
                option_a, option_b = incorrect_choice, correct_choice
                correct_label = "B"

            prompt = (
                f"Question: {question}\n\nA) {option_a}\nB) {option_b}\n\n"
                "Answer with only A or B:"
            )
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            inputs = {k: v.to(primary_device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = model(**inputs).logits[0, -1, :]

            token_a = tokenizer.encode("A", add_special_tokens=False)[0]
            token_b = tokenizer.encode("B", add_special_tokens=False)[0]
            probs = torch.softmax(logits[[token_a, token_b]], dim=0)
            prediction = "A" if probs[0] >= probs[1] else "B"

            results.append(
                {
                    "question": question,
                    "choice_a": option_a,
                    "choice_b": option_b,
                    "correct_label": correct_label,
                    "predicted": prediction,
                    "correct_prob": probs[0].item() if correct_label == "A" else probs[1].item(),
                    "incorrect_prob": probs[1].item() if correct_label == "A" else probs[0].item(),
                    "scale": scale,
                }
            )

    if not results:
        return {"stats": MCStats(0.0, 0.0, 0.0, 0), "details": []}

    accuracy = np.mean([r["predicted"] == r["correct_label"] for r in results])
    avg_correct_prob = np.mean([r["correct_prob"] for r in results])
    avg_incorrect_prob = np.mean([r["incorrect_prob"] for r in results])
    stats = MCStats(
        accuracy=float(accuracy),
        avg_correct_prob=float(avg_correct_prob),
        avg_incorrect_prob=float(avg_incorrect_prob),
        total=len(results),
    )
    return {"stats": stats, "details": results}


def evaluate_generation(
    model,
    tokenizer,
    items: Sequence[dict],
    *,
    layer_index: int,
    steering_vector: Optional[torch.Tensor],
    scale: float,
    generation_cfg: Dict,
    primary_device: torch.device,
    judge=None,
    semantic_judge=None,
) -> Dict:
    """Run open-ended generation with optional steering and judge evaluation."""
    steering = steering_vector if scale != 0 else None
    results = []

    # Get preset from config (default to simple format if not specified)
    preset = generation_cfg.get("preset", None)
    if preset:
        from src.prompts.truthfulqa_presets import format_prompt

    with steering_hook(model, layer_index, steering, scale=scale):
        for item in items:
            question = item["question"]

            # Format prompt using TruthfulQA preset if specified
            if preset:
                try:
                    prompt = format_prompt(question, preset=preset)
                except Exception as exc:
                    logger.warning("Failed to format with preset '%s': %s; falling back to simple format", preset, exc)
                    prompt = f"Question: {question}\nAnswer concisely in one sentence:"
            else:
                # Add one-sentence instruction to reduce verbosity
                prompt = f"Question: {question}\nAnswer concisely in one sentence:"
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=generation_cfg.get("max_length", 512),
            )
            inputs = {k: v.to(primary_device) for k, v in inputs.items()}

            is_gemma3 = _detect_gemma3(model)

            gen_kwargs = dict(
                max_new_tokens=generation_cfg.get("max_new_tokens", 80),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # Add stop sequences if provided
            stop_sequences = generation_cfg.get("stop_sequences", [])
            if stop_sequences and hasattr(tokenizer, "encode"):
                # Convert stop sequences to token ids for stopping criteria
                stop_token_ids = [
                    tokenizer.encode(seq, add_special_tokens=False)
                    for seq in stop_sequences
                ]
                # Flatten if single tokens
                stop_token_ids = [ids[0] if len(ids) == 1 else ids for ids in stop_token_ids]
                if stop_token_ids:
                    from transformers import StoppingCriteriaList, StoppingCriteria
                    class StopOnTokens(StoppingCriteria):
                        def __init__(self, stop_ids):
                            self.stop_ids = stop_ids
                        def __call__(self, input_ids, scores, **kwargs):
                            for stop_id in self.stop_ids:
                                if isinstance(stop_id, list):
                                    # Multi-token sequence
                                    if len(input_ids[0]) >= len(stop_id):
                                        if input_ids[0, -len(stop_id):].tolist() == stop_id:
                                            return True
                                else:
                                    # Single token
                                    if input_ids[0, -1] == stop_id:
                                        return True
                            return False
                    gen_kwargs["stopping_criteria"] = StoppingCriteriaList([StopOnTokens(stop_token_ids)])

            if not is_gemma3:
                gen_kwargs.update(
                    do_sample=True,
                    temperature=generation_cfg.get("temperature", 0.7),
                    top_p=generation_cfg.get("top_p", 0.9),
                    top_k=generation_cfg.get("top_k", 50),
                )
            else:
                gen_kwargs.update(do_sample=False)

            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)

            generated = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )

            # Prepare true answers with "I have no comment" added if not present
            true_answers = list(item.get("correct_answers") or [])
            if "I have no comment." not in true_answers:
                true_answers.append("I have no comment.")

            results.append(
                {
                    "question": question,
                    "generated": generated.strip(),
                    "best_answer": item.get("best_answer") or item["correct_answers"][0],
                    "true_answers": true_answers,
                    "incorrect_answers": item.get("incorrect_answers") or [],
                    "scale": scale,
                }
            )

    annotated = results
    if semantic_judge:
        annotated = semantic_judge.score_responses(annotated)
    if judge:
        annotated = judge.score_responses(annotated)

    stats = _summarize_generation(
        annotated,
        judge is not None,
        semantic_judge is not None,
    )
    return {"stats": stats, "details": annotated}


def _detect_gemma3(model) -> bool:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return False
    model_type = getattr(cfg, "model_type", "") or ""
    if "gemma-3" in model_type or "gemma3" in model_type:
        return True
    name_or_path = getattr(cfg, "_name_or_path", "") or ""
    return "gemma-3" in name_or_path.lower()


def _summarize_generation(
    results: Iterable[Dict],
    judged: bool,
    semantic_used: bool,
) -> GenerationStats:
    if not results:
        return GenerationStats(accuracy=None, semantic_mean=None, total=0)

    accuracy = None
    if judged:
        accuracy = float(np.mean([r.get("match", 0) for r in results]))

    semantic_mean = None
    if semantic_used:
        semantic_scores = [r.get("semantic_score") for r in results if r.get("semantic_score") is not None]
        if semantic_scores:
            semantic_mean = float(np.mean(semantic_scores))

    return GenerationStats(
        accuracy=accuracy,
        semantic_mean=semantic_mean,
        total=len(results),
    )
