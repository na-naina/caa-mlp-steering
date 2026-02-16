#!/usr/bin/env python3
"""Plot hyperparameter sweep results.

Usage:
    python scripts/plot_sweep.py <sweep_dir>
    python scripts/plot_sweep.py data/outputs/llama2/llama2_7b_chat/sweep_20260216_145758

Produces:
    <sweep_dir>/figures/sweep_mc_by_bn.png     - MC accuracy vs bottleneck dim
    <sweep_dir>/figures/sweep_mc_by_lr.png     - MC accuracy vs LR
    <sweep_dir>/figures/sweep_gap.png          - Train-test gap
    <sweep_dir>/figures/sweep_layers.png       - Layer comparison
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np


def load_results(sweep_dir: Path) -> list[dict]:
    p = sweep_dir / "phase1_results.json"
    if not p.exists():
        raise FileNotFoundError(f"No phase1_results.json in {sweep_dir}")
    with open(p) as f:
        return json.load(f)


def _bn_label(bn) -> str:
    return "fat" if bn is None else str(bn)


def _bn_sort_key(bn) -> int:
    return 999999 if bn is None else bn


def plot_mc_by_bottleneck(results: list[dict], fig_dir: Path) -> None:
    """MC accuracy vs bottleneck dim, one line per LR, faceted by layer."""
    layers = sorted(set(r["layer"] for r in results))
    lrs = sorted(set(r["lr"] for r in results))
    bns = sorted(set(r.get("bottleneck_dim") for r in results), key=_bn_sort_key)
    bn_labels = [_bn_label(b) for b in bns]

    fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 4), sharey=True)
    if len(layers) == 1:
        axes = [axes]

    for ax, layer in zip(axes, layers):
        baseline = None
        for r in results:
            if r["layer"] == layer:
                baseline = r.get("baselines", {}).get("baseline", {}).get("accuracy")
                if baseline is not None:
                    break

        for lr in lrs:
            accs = []
            for bn in bns:
                match = [
                    r for r in results
                    if r["layer"] == layer and r["lr"] == lr
                    and r.get("bottleneck_dim") == bn
                ]
                if match:
                    acc = match[0].get("mc_eval", {}).get("mlp_mc", {}).get("accuracy", 0)
                    accs.append(acc * 100)
                else:
                    accs.append(np.nan)
            ax.plot(range(len(bns)), accs, "o-", label=f"lr={lr:.0e}", markersize=6)

        if baseline is not None:
            ax.axhline(baseline * 100, color="gray", linestyle="--", alpha=0.7, label="baseline")

        ax.set_xticks(range(len(bns)))
        ax.set_xticklabels(bn_labels)
        ax.set_xlabel("Bottleneck dim")
        ax.set_title(f"Layer {layer}")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("MC Accuracy (%)")
    axes[-1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    fig.suptitle("MC Accuracy by Bottleneck Dim", fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "sweep_mc_by_bn.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig_dir / 'sweep_mc_by_bn.png'}")


def plot_mc_by_lr(results: list[dict], fig_dir: Path) -> None:
    """MC accuracy vs LR, one line per bottleneck dim, faceted by layer."""
    layers = sorted(set(r["layer"] for r in results))
    lrs = sorted(set(r["lr"] for r in results))
    bns = sorted(set(r.get("bottleneck_dim") for r in results), key=_bn_sort_key)

    fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 4), sharey=True)
    if len(layers) == 1:
        axes = [axes]

    for ax, layer in zip(axes, layers):
        baseline = None
        for r in results:
            if r["layer"] == layer:
                baseline = r.get("baselines", {}).get("baseline", {}).get("accuracy")
                if baseline is not None:
                    break

        for bn in bns:
            accs = []
            for lr in lrs:
                match = [
                    r for r in results
                    if r["layer"] == layer and r["lr"] == lr
                    and r.get("bottleneck_dim") == bn
                ]
                if match:
                    acc = match[0].get("mc_eval", {}).get("mlp_mc", {}).get("accuracy", 0)
                    accs.append(acc * 100)
                else:
                    accs.append(np.nan)
            ax.plot(range(len(lrs)), accs, "o-", label=f"bn={_bn_label(bn)}", markersize=6)

        if baseline is not None:
            ax.axhline(baseline * 100, color="gray", linestyle="--", alpha=0.7, label="baseline")

        ax.set_xticks(range(len(lrs)))
        ax.set_xticklabels([f"{lr:.0e}" for lr in lrs], fontsize=8)
        ax.set_xlabel("Learning Rate")
        ax.set_title(f"Layer {layer}")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("MC Accuracy (%)")
    axes[-1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    fig.suptitle("MC Accuracy by Learning Rate", fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "sweep_mc_by_lr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig_dir / 'sweep_mc_by_lr.png'}")


def plot_train_test_gap(results: list[dict], fig_dir: Path) -> None:
    """Train-test accuracy gap vs bottleneck dim."""
    layers = sorted(set(r["layer"] for r in results))
    lrs = sorted(set(r["lr"] for r in results))
    bns = sorted(set(r.get("bottleneck_dim") for r in results), key=_bn_sort_key)
    bn_labels = [_bn_label(b) for b in bns]

    fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 4), sharey=True)
    if len(layers) == 1:
        axes = [axes]

    for ax, layer in zip(axes, layers):
        for lr in lrs:
            gaps = []
            for bn in bns:
                match = [
                    r for r in results
                    if r["layer"] == layer and r["lr"] == lr
                    and r.get("bottleneck_dim") == bn
                ]
                if match:
                    r = match[0]
                    train_acc = r.get("mc_final_acc", 0) or 0
                    test_acc = r.get("mc_eval", {}).get("mlp_mc", {}).get("accuracy", 0)
                    gaps.append((train_acc - test_acc) * 100)
                else:
                    gaps.append(np.nan)
            ax.plot(range(len(bns)), gaps, "o-", label=f"lr={lr:.0e}", markersize=6)

        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xticks(range(len(bns)))
        ax.set_xticklabels(bn_labels)
        ax.set_xlabel("Bottleneck dim")
        ax.set_title(f"Layer {layer}")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Train - Test Gap (pp)")
    axes[-1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    fig.suptitle("Overfitting Gap by Architecture", fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "sweep_gap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig_dir / 'sweep_gap.png'}")


def plot_layer_comparison(results: list[dict], fig_dir: Path) -> None:
    """Bar chart: best MC accuracy per layer, colored by bottleneck dim."""
    layers = sorted(set(r["layer"] for r in results))

    # Find best config per layer
    best_per_layer = {}
    for layer in layers:
        layer_results = [r for r in results if r["layer"] == layer]
        best = max(
            layer_results,
            key=lambda r: r.get("mc_eval", {}).get("mlp_mc", {}).get("accuracy", 0),
        )
        best_per_layer[layer] = best

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(layers))
    width = 0.35

    # Best MC accuracy
    mc_accs = [
        best_per_layer[l].get("mc_eval", {}).get("mlp_mc", {}).get("accuracy", 0) * 100
        for l in layers
    ]
    baselines = [
        best_per_layer[l].get("baselines", {}).get("baseline", {}).get("accuracy", 0) * 100
        for l in layers
    ]
    caa_accs = [
        best_per_layer[l].get("baselines", {}).get("steered", {}).get("accuracy", 0) * 100
        for l in layers
    ]

    bars = ax.bar(x - width / 2, mc_accs, width, label="Best MLP MC", color="steelblue")
    ax.bar(x + width / 2, caa_accs, width, label="Raw CAA", color="lightcoral", alpha=0.7)

    # Add baseline line
    ax.axhline(baselines[0], color="gray", linestyle="--", alpha=0.7, label="No steering")

    # Annotate bars with config info
    for i, layer in enumerate(layers):
        r = best_per_layer[layer]
        bn = _bn_label(r.get("bottleneck_dim"))
        lr = r["lr"]
        gap = (r.get("mc_final_acc", 0) or 0) - r.get("mc_eval", {}).get("mlp_mc", {}).get("accuracy", 0)
        ax.annotate(
            f"bn={bn}\nlr={lr:.0e}\ngap={gap*100:.0f}%",
            xy=(i - width / 2, mc_accs[i]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color="steelblue",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {l}" for l in layers])
    ax.set_ylabel("MC Accuracy (%)")
    ax.set_title("Best Config per Layer", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(fig_dir / "sweep_layers.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig_dir / 'sweep_layers.png'}")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <sweep_dir> [sweep_dir2 ...]")
        sys.exit(1)

    for sweep_path in sys.argv[1:]:
        sweep_dir = Path(sweep_path)
        print(f"\nProcessing: {sweep_dir}")
        results = load_results(sweep_dir)
        print(f"  Loaded {len(results)} configs")

        fig_dir = sweep_dir / "figures"
        fig_dir.mkdir(exist_ok=True)

        plot_mc_by_bottleneck(results, fig_dir)
        plot_mc_by_lr(results, fig_dir)
        plot_train_test_gap(results, fig_dir)
        plot_layer_comparison(results, fig_dir)

        print(f"  All figures saved to {fig_dir}")


if __name__ == "__main__":
    main()
