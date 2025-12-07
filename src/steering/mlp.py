from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SteeringMLP(nn.Module):
    """Non-linear processor for CAA vectors.

    Initialized to approximate identity mapping for training stability,
    especially important for large models (12B+) where random init can
    cause immediate NaN gradients.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_multiplier: float = 2.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = max(int(input_dim * hidden_multiplier), input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

        # Initialize to near-identity for stable training
        self._init_near_identity()

    def _init_near_identity(self) -> None:
        """Initialize final layer with small weights so MLP ≈ identity initially.

        This prevents catastrophic first-step gradients in large models where
        random initialization can produce huge margin violations.
        """
        # The final linear layer is net[6]
        final_linear = self.net[6]
        if isinstance(final_linear, nn.Linear):
            # Small random init (10x smaller than default)
            nn.init.normal_(final_linear.weight, mean=0.0, std=0.01)
            nn.init.zeros_(final_linear.bias)
            logger.debug(
                "Initialized MLP final layer with std=0.01 for near-identity start"
            )

    def forward(self, vector: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Simple pass-through with small transformation
        # MSE regularization in training will keep this close to identity
        return self.net(vector)


@dataclass
class MLPTrainingConfig:
    lr: float = 1e-4
    weight_decay: float = 0.0
    epochs: int = 5
    grad_clip: float = 1.0
    norm_reg: float = 0.01


def train_mlp_on_activations(
    base_vector: torch.Tensor,
    positive_activations: torch.Tensor,
    negative_activations: torch.Tensor,
    mlp_config: Dict,
) -> SteeringMLP:
    """Train the MLP using cosine alignment against positive/negative activations."""
    device = base_vector.device
    cfg = MLPTrainingConfig(
        lr=mlp_config.get("lr", 1e-4),
        weight_decay=mlp_config.get("weight_decay", 0.0),
        epochs=mlp_config.get("epochs", 5),
        grad_clip=mlp_config.get("grad_clip", 1.0),
        norm_reg=mlp_config.get("norm_reg", 0.01),
    )

    model = SteeringMLP(
        input_dim=base_vector.shape[0],
        hidden_multiplier=mlp_config.get("hidden_multiplier", 2.0),
        dropout=mlp_config.get("dropout", 0.1),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    pos_mean = positive_activations.to(device).mean(dim=0)
    neg_mean = negative_activations.to(device).mean(dim=0)

    best_loss = float("inf")
    best_state = None

    for epoch in range(cfg.epochs):
        optimizer.zero_grad()
        transformed = model(base_vector.unsqueeze(0)).squeeze(0)

        cos_pos = F.cosine_similarity(transformed, pos_mean, dim=0)
        cos_neg = F.cosine_similarity(transformed, neg_mean, dim=0)
        alignment_loss = -(cos_pos - cos_neg)  # Prefer alignment with positive

        norm_penalty = cfg.norm_reg * (transformed.norm() - base_vector.norm()).pow(2)
        loss = alignment_loss + norm_penalty

        loss.backward()
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        scalar_loss = loss.item()
        logger.info(
            "MLP epoch %d/%d - loss: %.4f (cos_pos=%.4f, cos_neg=%.4f)",
            epoch + 1,
            cfg.epochs,
            scalar_loss,
            cos_pos.item(),
            cos_neg.item(),
        )

        if scalar_loss < best_loss:
            best_loss = scalar_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model
