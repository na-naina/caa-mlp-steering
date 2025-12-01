from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SteeringMLP(nn.Module):
    """Non-linear processor for CAA vectors."""

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
            nn.LayerNorm(input_dim),
        )

    def forward(self, vector: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        centered = vector - vector.mean(dim=-1, keepdim=True)
        transformed = self.net(centered)

        # Keep output magnitude comparable to the input steering vector to avoid
        # blowing up residuals on large hidden sizes.
        target_norm = vector.float().norm(dim=-1, keepdim=True).clamp(min=1e-6)
        out_norm = transformed.float().norm(dim=-1, keepdim=True).clamp(min=1e-6)
        scaled = transformed * (target_norm / out_norm).to(transformed.dtype)
        return scaled


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
