from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import torch

from src.steering.extract import compute_caa_vector

logger = logging.getLogger(__name__)


@dataclass
class VectorBank:
    base_vector: torch.Tensor
    vectors: List[torch.Tensor]
    indices: List[List[int]]

    def sample(self, rng: np.random.Generator) -> torch.Tensor:
        if not self.vectors:
            return self.base_vector
        choice = int(rng.integers(0, len(self.vectors)))
        return self.vectors[choice]

    def sample_interpolated(self, rng: np.random.Generator) -> torch.Tensor:
        """Sample a random convex combination of bank vectors.

        Draws Dirichlet weights over all bank vectors to produce a novel
        vector in the same subspace — effectively infinite input diversity
        at zero extraction cost.
        """
        if not self.vectors:
            return self.base_vector
        weights = rng.dirichlet(np.ones(len(self.vectors)))
        weights_t = torch.tensor(weights, dtype=self.vectors[0].dtype)
        stacked = torch.stack(self.vectors)  # (n_vectors, dim)
        return (weights_t @ stacked)  # (dim,)


def create_individual_bank(
    positive_activations: torch.Tensor,
    negative_activations: torch.Tensor,
) -> VectorBank:
    """Build a bank of individual per-example steering vectors.

    Each vector is (pos_i - neg_i) for a single example. No averaging,
    no subsets, no Dirichlet. Maximum diversity, minimal machinery.
    The base_vector is still the mean for use at inference time.
    """
    diffs = positive_activations - negative_activations  # (N, hidden_dim)
    base = diffs.mean(dim=0)
    vectors = [diffs[i] for i in range(len(diffs))]
    indices = [[i] for i in range(len(diffs))]

    logger.info(
        "Built individual vector bank: %d per-example vectors (no averaging)",
        len(vectors),
    )

    # Override sample_interpolated to just pick one random vector
    # (Dirichlet over 100 individual diffs would average them out)
    bank = VectorBank(base_vector=base, vectors=vectors, indices=indices)
    bank.sample_interpolated = bank.sample
    return bank


def create_noise_bank(
    base_vector: torch.Tensor,
    vector_bank: VectorBank,
    seed: int = 0,
) -> tuple[torch.Tensor, VectorBank]:
    """Replace real steering vectors with random noise of matching statistics.

    Ablation control: same shape, norm, and bank structure, but no real
    steering signal. If a noise-trained MLP doesn't improve truthfulness,
    it proves the extracted vectors carry meaningful information.
    """
    rng = torch.Generator().manual_seed(seed)
    dim = base_vector.shape[0]
    base_norm = base_vector.norm().item()

    # Random base vector with matching norm
    noise_base = torch.randn(dim, generator=rng, dtype=base_vector.dtype)
    noise_base = noise_base * (base_norm / noise_base.norm().item())

    # Random bank vectors with matching norms
    noise_vectors = []
    for v in vector_bank.vectors:
        nv = torch.randn(dim, generator=rng, dtype=v.dtype)
        nv = nv * (v.norm().item() / nv.norm().item())
        noise_vectors.append(nv)

    noise_bank = VectorBank(
        base_vector=noise_base,
        vectors=noise_vectors,
        indices=vector_bank.indices,
    )

    logger.info(
        "Created noise bank: base_norm=%.4f, %d vectors (ablation control)",
        base_norm, len(noise_vectors),
    )
    return noise_base, noise_bank


class VectorBankBuilder:
    """Construct a bank of steering vectors from cached activations."""

    def __init__(
        self,
        positive_activations: torch.Tensor,
        negative_activations: torch.Tensor,
        *,
        normalize: bool = True,
        seed: int = 0,
    ) -> None:
        if positive_activations.shape != negative_activations.shape:
            raise ValueError("Positive/negative activations must have same shape")

        self.pos = positive_activations
        self.neg = negative_activations
        self.normalize = normalize
        self.rng = np.random.default_rng(seed)

    def build(
        self,
        *,
        num_vectors: int,
        sample_size_range: Sequence[int],
    ) -> VectorBank:
        if num_vectors < 0:
            raise ValueError("num_vectors must be non-negative")

        if len(sample_size_range) == 0:
            raise ValueError("sample_size_range cannot be empty")

        min_size = min(sample_size_range)
        max_size = max(sample_size_range)

        if min_size <= 0:
            raise ValueError("sample sizes must be positive")

        total_samples = self.pos.shape[0]
        if min_size > total_samples:
            logger.warning(
                "Requested min sample size %d exceeds available activations (%d); "
                "clamping to %d",
                min_size,
                total_samples,
                total_samples,
            )
            min_size = total_samples
        if max_size > total_samples:
            logger.warning(
                "Requested max sample size %d exceeds available activations (%d); "
                "clamping to %d",
                max_size,
                total_samples,
                total_samples,
            )
            max_size = total_samples

        sizes = self._resolve_sample_sizes(num_vectors, min_size, max_size)

        base_vector = compute_caa_vector(self.pos, self.neg, normalize=self.normalize)
        vectors: List[torch.Tensor] = []
        subsets: List[List[int]] = []

        for vec_idx, subset_size in enumerate(sizes):
            subset = self._sample_indices(subset_size, total_samples)
            subset_tensor = torch.tensor(subset, dtype=torch.long)
            pos_subset = self.pos[subset_tensor]
            neg_subset = self.neg[subset_tensor]
            vector = compute_caa_vector(
                pos_subset,
                neg_subset,
                normalize=self.normalize,
            )
            vectors.append(vector)
            subsets.append(subset)
            logger.debug(
                "Generated vector %d using %d samples (indices=%s)",
                vec_idx,
                subset_size,
                subset,
            )

        logger.info(
            "Built vector bank with base vector and %d sampled vectors (range %d-%d)",
            len(vectors),
            min_size,
            max_size,
        )
        return VectorBank(base_vector=base_vector, vectors=vectors, indices=subsets)

    def _sample_indices(self, size: int, total: int) -> List[int]:
        if size >= total:
            return list(range(total))
        return self.rng.choice(total, size=size, replace=False).tolist()

    def _resolve_sample_sizes(
        self,
        num_vectors: int,
        min_size: int,
        max_size: int,
    ) -> List[int]:
        if num_vectors == 0:
            return []
        if min_size == max_size:
            return [min_size for _ in range(num_vectors)]

        sizes = self.rng.integers(min_size, max_size + 1, size=num_vectors)
        return [int(s) for s in sizes]
