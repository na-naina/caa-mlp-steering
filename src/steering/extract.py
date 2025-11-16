from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Callable, Iterable, List

import torch
import torch.nn.functional as F

from src.models.loader import LoadedModel

logger = logging.getLogger(__name__)


def _get_decoder_layers(model) -> list:
    # Standard text-only architecture (Gemma, Gemma2, Gemma3ForCausalLM, etc.)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # Multimodal architecture (Gemma3ForConditionalGeneration - 12B+)
    # Has model.language_model.layers structure
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        if hasattr(model.model.language_model, "layers"):
            return model.model.language_model.layers
    # GPT-style architecture
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Unsupported model architecture for activation extraction")


@contextmanager
def _activation_hook(layer, callback: Callable[[torch.Tensor], None]):
    def hook(_module, _input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        callback(hidden.detach())

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


class ActivationExtractor:
    """Capture and aggregate residual stream activations from a decoder layer."""

    def __init__(
        self,
        loaded: LoadedModel,
        layer_index: int,
        *,
        max_length: int = 512,
        batch_size: int = 8,
    ) -> None:
        self.loaded = loaded
        self.layer_index = layer_index
        self.max_length = max_length
        self.batch_size = batch_size

        layers = _get_decoder_layers(self.loaded.model)
        if layer_index < 0 or layer_index >= len(layers):
            raise IndexError(
                f"Layer index {layer_index} out of bounds for {len(layers)} layers"
            )
        self.layer = layers[layer_index]

    def _run_batch(self, texts: List[str]) -> torch.Tensor:
        activations: List[torch.Tensor] = []
        attention_mask: torch.Tensor | None = None

        def collect(hidden: torch.Tensor) -> None:
            # Upcast to fp32 to avoid bfloat16 numerical issues
            activations.append(hidden.float())

        with _activation_hook(self.layer, collect):
            encoded = self.loaded.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            encoded = {k: v.to(self.loaded.primary_device) for k, v in encoded.items()}
            attention_mask = encoded.get("attention_mask")
            # Disable autocast during extraction to prevent precision issues
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                _ = self.loaded.model(**encoded)

        if not activations:
            raise RuntimeError("Failed to capture activations for batch")
        if attention_mask is None:
            raise RuntimeError("Attention mask missing during activation capture")

        hidden = activations[0]

        # Check for NaN/Inf in raw activations
        if torch.isnan(hidden).any() or torch.isinf(hidden).any():
            # Find which examples in the batch have issues
            for i, text in enumerate(texts):
                example_acts = hidden[i]
                if torch.isnan(example_acts).any() or torch.isinf(example_acts).any():
                    nan_count = torch.isnan(example_acts).sum().item()
                    inf_count = torch.isinf(example_acts).sum().item()
                    logger.error(
                        f"Invalid activations in batch example {i}: "
                        f"NaN count={nan_count}, Inf count={inf_count}"
                    )
                    logger.error(f"Problematic text (first 200 chars): {text[:200]}")
            raise ValueError("Activations contain NaN or Inf values")

        # Compute mean over non-padding tokens only
        mask = attention_mask.to(hidden.device).unsqueeze(-1).float()
        token_counts = mask.sum(dim=1).clamp(min=1.0)
        masked_sum = (hidden * mask).sum(dim=1)
        mean_hidden = masked_sum / token_counts  # batch, hidden_dim
        return mean_hidden.cpu()

    def collect_mean_activations(self, texts: Iterable[str]) -> tuple[torch.Tensor, list[int]]:
        """
        Collect activations and return tuple of (activations, valid_indices).

        Returns:
            activations: Tensor of shape (num_valid, hidden_dim)
            valid_indices: List of original indices that produced valid activations
        """
        text_list = list(texts)
        batches = [
            text_list[i : i + self.batch_size]
            for i in range(0, len(text_list), self.batch_size)
        ]

        all_activations = []
        valid_indices = []
        current_idx = 0

        for batch in batches:
            try:
                batch_acts = self._run_batch(batch)
                all_activations.append(batch_acts)
                # All examples in batch were valid
                valid_indices.extend(range(current_idx, current_idx + len(batch)))
                current_idx += len(batch)
            except ValueError as e:
                if "NaN or Inf" in str(e):
                    logger.warning(f"NaN/Inf in batch starting at index {current_idx}, processing individually")
                    # Process examples one at a time to identify valid ones
                    for i, text in enumerate(batch):
                        try:
                            result = self._run_batch([text])
                            all_activations.append(result)
                            valid_indices.append(current_idx + i)
                        except ValueError as e_single:
                            if "NaN or Inf" in str(e_single):
                                logger.warning(f"Skipping index {current_idx + i} due to NaN/Inf: {text[:100]}...")
                            else:
                                raise
                    current_idx += len(batch)
                else:
                    raise

        if not all_activations:
            raise RuntimeError("All examples produced NaN/Inf - no valid activations collected")

        activations = torch.cat(all_activations, dim=0)
        return activations, valid_indices


def compute_caa_vector(
    activations_positive: torch.Tensor,
    activations_negative: torch.Tensor,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute mean difference vector between positive and negative activations."""
    if activations_positive.shape != activations_negative.shape:
        raise ValueError("Positive and negative activations must share the same shape")

    # Check input activations before computing means
    if torch.isnan(activations_positive).any() or torch.isinf(activations_positive).any():
        nan_count = torch.isnan(activations_positive).sum().item()
        inf_count = torch.isinf(activations_positive).sum().item()
        logger.error(f"Positive activations contain NaN ({nan_count}) or Inf ({inf_count})")
        # Log statistics about the problematic activations
        logger.error(f"Positive acts shape: {activations_positive.shape}")
        logger.error(f"Positive acts min: {activations_positive[~torch.isnan(activations_positive) & ~torch.isinf(activations_positive)].min().item():.4e}")
        logger.error(f"Positive acts max: {activations_positive[~torch.isnan(activations_positive) & ~torch.isinf(activations_positive)].max().item():.4e}")
        raise ValueError("Positive activations contain NaN or Inf")

    if torch.isnan(activations_negative).any() or torch.isinf(activations_negative).any():
        nan_count = torch.isnan(activations_negative).sum().item()
        inf_count = torch.isinf(activations_negative).sum().item()
        logger.error(f"Negative activations contain NaN ({nan_count}) or Inf ({inf_count})")
        # Log statistics about the problematic activations
        logger.error(f"Negative acts shape: {activations_negative.shape}")
        logger.error(f"Negative acts min: {activations_negative[~torch.isnan(activations_negative) & ~torch.isinf(activations_negative)].min().item():.4e}")
        logger.error(f"Negative acts max: {activations_negative[~torch.isnan(activations_negative) & ~torch.isinf(activations_negative)].max().item():.4e}")
        raise ValueError("Negative activations contain NaN or Inf")

    vector = activations_positive.mean(dim=0) - activations_negative.mean(dim=0)

    # Check for NaN/Inf before normalization
    if torch.isnan(vector).any() or torch.isinf(vector).any():
        logger.error("CAA vector contains NaN or Inf after computing mean difference")
        logger.error(f"Positive mean min/max: {activations_positive.mean(dim=0).min():.4e} / {activations_positive.mean(dim=0).max():.4e}")
        logger.error(f"Negative mean min/max: {activations_negative.mean(dim=0).min():.4e} / {activations_negative.mean(dim=0).max():.4e}")
        raise ValueError("Invalid CAA vector: contains NaN or Inf")

    if normalize:
        norm = vector.norm()
        if norm < 1e-8:
            logger.warning(
                f"CAA vector has near-zero norm ({norm:.2e}), skipping normalization. "
                "This may indicate the model doesn't differentiate positive/negative at this layer."
            )
            return vector
        vector = F.normalize(vector, dim=0)

    return vector
