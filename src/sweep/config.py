"""Sweep grid definition and expansion."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, List


@dataclass
class SweepConfig:
    """Definition of a hyperparameter sweep grid."""

    layers: List[int]
    learning_rates: List[float]
    mse_regs: List[float]
    top_k: int = 5
    mc_only_variants: List[str] = field(
        default_factory=lambda: ["mlp_mc", "mlp_gen"]
    )

    @property
    def total_configs(self) -> int:
        return len(self.layers) * len(self.learning_rates) * len(self.mse_regs)

    def configs_for_layer(self, layer: int) -> List[Dict[str, Any]]:
        """Return all HP combos for a given layer."""
        combos = []
        for lr, reg in product(self.learning_rates, self.mse_regs):
            combos.append({
                "layer": layer,
                "lr": lr,
                "mse_reg": reg,
                "combo_id": combo_dir_name(lr, reg),
            })
        return combos


def combo_dir_name(lr: float, mse_reg: float) -> str:
    """Generate a human-readable directory name for an HP combo."""
    return f"lr{lr:.0e}_reg{mse_reg:.0e}"
