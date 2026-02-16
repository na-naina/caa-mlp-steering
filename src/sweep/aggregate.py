"""Results collection, ranking, and table formatting for sweeps."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def load_phase1_results(sweep_dir: Path) -> List[Dict[str, Any]]:
    """Load Phase 1 results from the aggregated file or individual mc_eval files."""
    agg = sweep_dir / "phase1_results.json"
    if agg.exists():
        with open(agg) as f:
            return json.load(f)

    # Fallback: reconstruct from individual files
    results = []
    for mc_file in sorted(sweep_dir.rglob("mc_eval.json")):
        with open(mc_file) as f:
            results.append(json.load(f))
    return results


def rank_configs(
    results: List[Dict[str, Any]],
    variant: str = "mlp_mc",
    metric: str = "accuracy",
) -> List[Dict[str, Any]]:
    """Rank configs by MC accuracy for a given variant, descending."""

    def _key(r: dict) -> float:
        return r.get("mc_eval", {}).get(variant, {}).get(metric, -1.0)

    ranked = sorted(results, key=_key, reverse=True)
    for i, r in enumerate(ranked):
        r["rank"] = i + 1
    return ranked


def select_top_k(
    results: List[Dict[str, Any]],
    top_k: int = 5,
    variant: str = "mlp_mc",
) -> List[Dict[str, Any]]:
    """Select top-K configs for Phase 2.  Excludes NaN / invalid training."""
    valid = [r for r in results if r.get("mc_train_valid", False)]
    ranked = rank_configs(valid, variant=variant)
    return ranked[:top_k]


def print_results_table(results: List[Dict[str, Any]]) -> None:
    """Print a human-readable results table."""
    header = (
        f"{'Rank':>4}  {'Layer':>5}  {'LR':>10}  {'MSE Reg':>10}  "
        f"{'MC_mc%':>8}  {'MC_gen%':>8}  {'BL%':>7}  {'CAA%':>7}  {'OK':>3}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        mc_mc = r.get("mc_eval", {}).get("mlp_mc", {}).get("accuracy", 0) * 100
        mc_gen = r.get("mc_eval", {}).get("mlp_gen", {}).get("accuracy", 0) * 100
        bl = r.get("baselines", {}).get("baseline", {}).get("accuracy", 0) * 100
        caa = r.get("baselines", {}).get("steered", {}).get("accuracy", 0) * 100
        ok = "Y" if r.get("mc_train_valid") else "N"
        print(
            f"{r.get('rank', '-'):>4}  {r['layer']:>5}  {r['lr']:>10.0e}  "
            f"{r['mse_reg']:>10.0e}  {mc_mc:>7.1f}%  {mc_gen:>7.1f}%  "
            f"{bl:>6.1f}%  {caa:>6.1f}%  {ok:>3}"
        )
