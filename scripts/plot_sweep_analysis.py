#!/usr/bin/env python3
"""Comprehensive sweep analysis plots across all sweep runs.

Usage:
    python scripts/plot_sweep_analysis.py

Reads all phase1_results.json from sweep directories and produces
a multi-panel figure showing trends across HP dimensions.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

SWEEP_BASE = Path("data/outputs/sweep_results/data/outputs/llama2/llama2_7b_chat")
OUT_DIR = Path("paper/figures/sweep")


def load_all_results() -> list[dict]:
    """Load and merge all sweep results, normalizing fields."""
    all_results = []
    for sweep_dir in sorted(SWEEP_BASE.glob("sweep_*")):
        p = sweep_dir / "phase1_results.json"
        if not p.exists():
            continue
        with open(p) as f:
            results = json.load(f)
        for r in results:
            r.setdefault("bottleneck_dim", None)
            r["_sweep"] = sweep_dir.name
        all_results.extend(results)
    print(f"Loaded {len(all_results)} total configs from {len(set(r['_sweep'] for r in all_results))} sweeps")
    return all_results


def bn_label(bn) -> str:
    return "fat" if bn is None else str(bn)


def fig_overfitting_vs_bottleneck(results: list[dict], axes):
    """Left panel: Train-test gap vs bottleneck dim, showing overfitting cliff."""
    # Use only the bottleneck sweep (sweep with bn data and 408 test items)
    bn_results = [r for r in results if r["mc_eval"]["mlp_mc"]["total"] == 408
                  and r.get("bottleneck_dim") is not None or r.get("bottleneck_dim") is None]
    # Actually filter to sweep_2 which has all bn variants
    bn_results = [r for r in results if "145758" in r["_sweep"]]
    if not bn_results:
        return

    ax_gap, ax_acc = axes

    layers = sorted(set(r["layer"] for r in bn_results))
    lrs = sorted(set(r["lr"] for r in bn_results))
    bns_raw = sorted(set(r.get("bottleneck_dim") for r in bn_results),
                     key=lambda x: x if x is not None else 999999)
    bn_positions = list(range(len(bns_raw)))
    bn_labels = [bn_label(b) for b in bns_raw]

    # Color by layer, line style by LR
    layer_colors = {8: "#2196F3", 12: "#FF9800", 16: "#4CAF50", 20: "#9C27B0"}
    lr_styles = {lrs[i]: ["-", "--", ":", "-."][i] for i in range(len(lrs))}

    # Plot train-test gap
    for layer in layers:
        for lr in lrs:
            gaps = []
            test_accs = []
            for bn in bns_raw:
                match = [r for r in bn_results
                         if r["layer"] == layer and r["lr"] == lr
                         and r.get("bottleneck_dim") == bn]
                if match:
                    r = match[0]
                    train = (r.get("mc_final_acc") or 0)
                    test = r["mc_eval"]["mlp_mc"]["accuracy"]
                    gaps.append((train - test) * 100)
                    test_accs.append(test * 100)
                else:
                    gaps.append(np.nan)
                    test_accs.append(np.nan)

            label = f"L{layer} lr={lr:.0e}" if lr == lrs[-1] else None  # label once
            ax_gap.plot(bn_positions, gaps, marker="o", markersize=5,
                        color=layer_colors.get(layer, "gray"),
                        linestyle=lr_styles[lr], alpha=0.8,
                        label=f"L{layer} lr={lr:.0e}")
            ax_acc.plot(bn_positions, test_accs, marker="o", markersize=5,
                        color=layer_colors.get(layer, "gray"),
                        linestyle=lr_styles[lr], alpha=0.8)

    ax_gap.axhline(0, color="gray", linestyle="-", alpha=0.3)
    ax_gap.set_xticks(bn_positions)
    ax_gap.set_xticklabels(bn_labels)
    ax_gap.set_xlabel("Bottleneck Dimension")
    ax_gap.set_ylabel("Train - Test Gap (pp)")
    ax_gap.set_title("Overfitting by Architecture", fontweight="bold", fontsize=11)
    ax_gap.grid(True, alpha=0.2)

    baseline = bn_results[0]["baselines"]["baseline"]["accuracy"] * 100
    ax_acc.axhline(baseline, color="gray", linestyle="--", alpha=0.6, label="baseline")
    ax_acc.set_xticks(bn_positions)
    ax_acc.set_xticklabels(bn_labels)
    ax_acc.set_xlabel("Bottleneck Dimension")
    ax_acc.set_ylabel("Test MC Accuracy (%)")
    ax_acc.set_title("Test Accuracy by Architecture", fontweight="bold", fontsize=11)
    ax_acc.grid(True, alpha=0.2)


def fig_lr_effect(results: list[dict], axes):
    """Middle panel: Test accuracy vs LR for bn=16 across layers."""
    # Combine sweep 2 and 3 for bn=16 data
    bn16 = [r for r in results
            if r.get("bottleneck_dim") == 16 and r["mc_eval"]["mlp_mc"]["total"] == 408]
    if not bn16:
        return

    ax_acc, ax_gap = axes

    layers = sorted(set(r["layer"] for r in bn16))
    lrs = sorted(set(r["lr"] for r in bn16))

    layer_colors = {8: "#2196F3", 12: "#FF9800", 16: "#4CAF50", 20: "#9C27B0"}

    for layer in layers:
        accs = []
        gaps = []
        valid_lrs = []
        for lr in lrs:
            match = [r for r in bn16 if r["layer"] == layer and r["lr"] == lr]
            if match:
                r = match[0]
                test = r["mc_eval"]["mlp_mc"]["accuracy"]
                train = r.get("mc_final_acc") or 0
                accs.append(test * 100)
                gaps.append((train - test) * 100)
                valid_lrs.append(lr)

        if valid_lrs:
            ax_acc.plot(range(len(valid_lrs)), accs, "o-", markersize=6,
                        color=layer_colors.get(layer, "gray"),
                        label=f"Layer {layer}")
            ax_gap.plot(range(len(valid_lrs)), gaps, "o-", markersize=6,
                        color=layer_colors.get(layer, "gray"),
                        label=f"Layer {layer}")

    baseline = bn16[0]["baselines"]["baseline"]["accuracy"] * 100
    ax_acc.axhline(baseline, color="gray", linestyle="--", alpha=0.6, label="baseline")
    lr_labels = [f"{lr:.0e}" for lr in lrs]
    ax_acc.set_xticks(range(len(lrs)))
    ax_acc.set_xticklabels(lr_labels, fontsize=8)
    ax_acc.set_xlabel("Learning Rate")
    ax_acc.set_ylabel("Test MC Accuracy (%)")
    ax_acc.set_title("LR Effect (bn=16)", fontweight="bold", fontsize=11)
    ax_acc.grid(True, alpha=0.2)

    ax_gap.axhline(0, color="gray", linestyle="-", alpha=0.3)
    ax_gap.set_xticks(range(len(lrs)))
    ax_gap.set_xticklabels(lr_labels, fontsize=8)
    ax_gap.set_xlabel("Learning Rate")
    ax_gap.set_ylabel("Train - Test Gap (pp)")
    ax_gap.set_title("Overfitting vs LR (bn=16)", fontweight="bold", fontsize=11)
    ax_gap.grid(True, alpha=0.2)


def fig_layer_comparison(results: list[dict], ax):
    """Right panel: Best test accuracy per layer across all sweeps."""
    # Use 408-test-item sweeps only for fair comparison
    fair = [r for r in results if r["mc_eval"]["mlp_mc"]["total"] == 408]
    if not fair:
        return

    layers = sorted(set(r["layer"] for r in fair))
    baseline = fair[0]["baselines"]["baseline"]["accuracy"] * 100

    # Best bottleneck per layer
    best_bn = {}
    best_fat = {}
    for layer in layers:
        layer_bn = [r for r in fair if r["layer"] == layer and r.get("bottleneck_dim") is not None]
        layer_fat = [r for r in fair if r["layer"] == layer and r.get("bottleneck_dim") is None]
        if layer_bn:
            best = max(layer_bn, key=lambda r: r["mc_eval"]["mlp_mc"]["accuracy"])
            best_bn[layer] = best
        if layer_fat:
            best = max(layer_fat, key=lambda r: r["mc_eval"]["mlp_mc"]["accuracy"])
            best_fat[layer] = best

    x = np.arange(len(layers))
    width = 0.35

    bn_accs = [best_bn[l]["mc_eval"]["mlp_mc"]["accuracy"] * 100 if l in best_bn else 0 for l in layers]
    fat_accs = [best_fat[l]["mc_eval"]["mlp_mc"]["accuracy"] * 100 if l in best_fat else 0 for l in layers]
    bn_gaps = [(best_bn[l].get("mc_final_acc", 0) or 0) - best_bn[l]["mc_eval"]["mlp_mc"]["accuracy"]
               if l in best_bn else 0 for l in layers]

    bars_bn = ax.bar(x - width/2, bn_accs, width, label="Best bottleneck", color="#2196F3", alpha=0.85)
    bars_fat = ax.bar(x + width/2, fat_accs, width, label="Best fat MLP", color="#FF5722", alpha=0.65)
    ax.axhline(baseline, color="gray", linestyle="--", alpha=0.6, label=f"baseline ({baseline:.1f}%)")

    # Annotate bottleneck bars with config
    for i, layer in enumerate(layers):
        if layer in best_bn:
            r = best_bn[layer]
            bn = r["bottleneck_dim"]
            lr = r["lr"]
            gap = ((r.get("mc_final_acc") or 0) - r["mc_eval"]["mlp_mc"]["accuracy"]) * 100
            ax.annotate(f"bn={bn}\nlr={lr:.0e}\ngap={gap:.0f}pp",
                        xy=(i - width/2, bn_accs[i]),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", fontsize=6.5, color="#1565C0")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {l}" for l in layers])
    ax.set_ylabel("Test MC Accuracy (%)")
    ax.set_title("Best Config per Layer", fontweight="bold", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2, axis="y")


def fig_heatmap(results: list[dict], ax):
    """Heatmap: Test accuracy for bn=16 configs (layer x LR)."""
    bn16 = [r for r in results
            if r.get("bottleneck_dim") == 16 and r["mc_eval"]["mlp_mc"]["total"] == 408]
    if not bn16:
        return

    layers = sorted(set(r["layer"] for r in bn16))
    lrs = sorted(set(r["lr"] for r in bn16))
    baseline = bn16[0]["baselines"]["baseline"]["accuracy"]

    grid = np.full((len(layers), len(lrs)), np.nan)
    for r in bn16:
        li = layers.index(r["layer"])
        lri = lrs.index(r["lr"])
        acc = r["mc_eval"]["mlp_mc"]["accuracy"]
        # Keep best if duplicates
        if np.isnan(grid[li, lri]) or acc > grid[li, lri]:
            grid[li, lri] = acc

    # Show as percentage points above baseline
    delta = (grid - baseline) * 100

    im = ax.imshow(delta, cmap="RdYlGn", aspect="auto",
                   vmin=-5, vmax=15)
    ax.set_xticks(range(len(lrs)))
    ax.set_xticklabels([f"{lr:.0e}" for lr in lrs], fontsize=8)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{l}" for l in layers])
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Layer")
    ax.set_title("bn=16: Accuracy vs Baseline (pp)", fontweight="bold", fontsize=11)

    # Annotate cells
    for i in range(len(layers)):
        for j in range(len(lrs)):
            val = delta[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > 8 else "black"
                ax.text(j, i, f"{val:+.1f}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=color)

    plt.colorbar(im, ax=ax, label="pp above baseline", shrink=0.8)


def fig_fat_vs_bottleneck_scatter(results: list[dict], ax):
    """Scatter: train acc vs test acc, colored by architecture type."""
    fair = [r for r in results if r["mc_eval"]["mlp_mc"]["total"] == 408]
    if not fair:
        return

    for r in fair:
        train = (r.get("mc_final_acc") or 0) * 100
        test = r["mc_eval"]["mlp_mc"]["accuracy"] * 100
        bn = r.get("bottleneck_dim")
        if bn is None:
            ax.scatter(train, test, c="#FF5722", alpha=0.5, s=30, edgecolors="none")
        elif bn == 16:
            ax.scatter(train, test, c="#2196F3", alpha=0.7, s=40, edgecolors="none")
        elif bn == 4:
            ax.scatter(train, test, c="#81D4FA", alpha=0.5, s=25, edgecolors="none")
        elif bn == 64:
            ax.scatter(train, test, c="#0D47A1", alpha=0.5, s=25, edgecolors="none")

    # Identity line (no overfitting)
    ax.plot([40, 105], [40, 105], "k--", alpha=0.2, linewidth=1)

    # Baseline
    baseline = fair[0]["baselines"]["baseline"]["accuracy"] * 100
    ax.axhline(baseline, color="gray", linestyle=":", alpha=0.4)
    ax.axvline(baseline, color="gray", linestyle=":", alpha=0.4)

    # Legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#FF5722", markersize=8, label="fat MLP"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2196F3", markersize=8, label="bn=16"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#81D4FA", markersize=7, label="bn=4"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#0D47A1", markersize=7, label="bn=64"),
    ]
    ax.legend(handles=handles, fontsize=7, loc="lower right")

    ax.set_xlabel("Train MC Accuracy (%)")
    ax.set_ylabel("Test MC Accuracy (%)")
    ax.set_title("Train vs Test (overfitting view)", fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(40, 105)
    ax.set_ylim(40, 65)


def main():
    results = load_all_results()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Figure 1: 3x2 comprehensive dashboard ----
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    fig.suptitle("Hyperparameter Sweep Analysis — Llama-2-7B-chat", fontsize=14, fontweight="bold", y=0.98)

    fig_overfitting_vs_bottleneck(results, (axes[0, 0], axes[0, 1]))
    fig_lr_effect(results, (axes[1, 0], axes[1, 1]))
    fig_fat_vs_bottleneck_scatter(results, axes[2, 0])
    fig_heatmap(results, axes[2, 1])

    # Shared legend for top rows
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles, labels, fontsize=6, loc="upper left", ncol=2)

    handles2, labels2 = axes[1, 0].get_legend_handles_labels()
    if handles2:
        axes[1, 0].legend(handles2, labels2, fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_DIR / "sweep_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'sweep_dashboard.png'}")

    # ---- Figure 2: Layer comparison bar chart ----
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    fig_layer_comparison(results, ax2)
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "sweep_layer_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved {OUT_DIR / 'sweep_layer_comparison.png'}")


if __name__ == "__main__":
    main()
