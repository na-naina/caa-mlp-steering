#!/usr/bin/env python3
"""Generate clean, conference-ready figures for the paper.

Style guidelines:
- Single-column figures: 3.3" wide (for 7.7cm ACL column)
- Two-panel figures: 6.7" wide (full text width)
- Grayscale-readable: distinct markers/line styles, not color-dependent
- Muted palette
- No gratuitous reference lines (e.g., no "RaLFiT SOTA" dashed line in every plot)
- Consistent fonts and sizes
- Tight layout, minimal gridlines
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

OUTDIR = Path(__file__).parent.parent / "paper" / "figures" / "paper"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Unified style
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "grid.linewidth": 0.4,
})

# Muted color palette — grayscale-distinguishable via markers/lines
C_OURS = "#1f77b4"      # blue (primary — "our method")
C_BASELINE = "#7f7f7f"  # gray (baselines / controls)
C_TRUTH = "#2ca02c"     # green (truth metric)
C_INFO = "#d62728"      # red (info metric)
C_TI = "#ff7f0e"        # orange (composite T*I metric)
C_NOISE = "#ff9896"     # light red (noise control)

COLUMN_W = 3.3
DOUBLE_W = 6.7


def save_both(fig, name):
    """Save PDF and PNG."""
    fig.savefig(OUTDIR / f"{name}.pdf")
    fig.savefig(OUTDIR / f"{name}.png")
    print(f"  Saved: {name}")


# ---------------------------------------------------------------
# 1. NOISE ABLATION — primary ablation, shows CAA init matters
# ---------------------------------------------------------------
def plot_noise_ablation():
    fig, ax = plt.subplots(figsize=(COLUMN_W, 2.0))

    # Numbers: baseline (no steering), noise control, our method
    # Baseline: 54.56% (RaLFiT-reported LLaMA-2-7B-Chat unsteered)
    # Noise ablation: 60.5% (single seed, seed 42)
    # MAST: 77.9 ± 2.7 (10-seed multi-seed mean reported in Table 1)
    labels = ["Baseline\n(no steering)", "MAST-Noise\n(ablation)", "MAST"]
    values = [54.6, 60.5, 77.9]
    errors = [None, None, 2.7]
    colors = [C_BASELINE, C_NOISE, C_OURS]
    hatches = ["", "///", ""]

    bars = ax.bar(labels, values, color=colors, edgecolor="black",
                  linewidth=0.7, hatch=hatches, width=0.65)

    # Error bar on ours
    if errors[2]:
        ax.errorbar(2, values[2], yerr=errors[2], fmt="none",
                    ecolor="black", capsize=4, linewidth=1)

    # Value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        err_str = f"$\\pm${errors[i]}" if errors[i] else ""
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val:.1f}{err_str}", ha="center", va="bottom",
                fontsize=8, fontweight="bold")

    ax.set_ylabel(r"T$\times$I (%)")
    ax.set_ylim(0, 90)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.grid(axis="y")

    fig.tight_layout()
    save_both(fig, "noise_ablation")
    plt.close(fig)


# ---------------------------------------------------------------
# 2. BOTTLENECK DIMENSION SWEEP
# ---------------------------------------------------------------
def plot_bottleneck():
    fig, ax = plt.subplots(figsize=(COLUMN_W, 2.0))

    dims = [4, 8, 32, 64]
    truth = [79.7, 83.1, 86.5, 86.5]
    info = [96.6, 97.1, 89.7, 75.2]
    ti = [76.5, 80.4, 76.5, 62.3]

    ax.plot(dims, truth, "s--", color=C_TRUTH, linewidth=1.3,
            markersize=6, label="Truth %", alpha=0.85)
    ax.plot(dims, info, "^--", color=C_INFO, linewidth=1.3,
            markersize=6, label="Info %", alpha=0.85)
    ax.plot(dims, ti, "o-", color=C_TI, linewidth=2.0,
            markersize=7, label=r"T$\times$I %", zorder=5)

    # Annotate best point
    best_idx = ti.index(max(ti))
    ax.annotate(f"{ti[best_idx]:.1f}%", xy=(dims[best_idx], ti[best_idx]),
                xytext=(6, -10), textcoords="offset points",
                fontsize=8, fontweight="bold", color=C_TI)

    # remove literal \% issue in annotation

    ax.set_xlabel(r"Bottleneck dimension $k$")
    ax.set_ylabel("Score (%)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_ylim(58, 102)
    ax.legend(loc="lower left", frameon=True, fancybox=False, edgecolor="0.5")

    fig.tight_layout()
    save_both(fig, "bottleneck_sweep")
    plt.close(fig)


# ---------------------------------------------------------------
# 3. SCALE SWEEP — two panels: performance vs scale & refusal rate
# ---------------------------------------------------------------
def plot_scale_sweep():
    fig, ax = plt.subplots(figsize=(COLUMN_W, 2.0))

    scales = [0.80, 0.90, 0.95, 1.00, 1.10, 1.20]
    truth = [82.6, 85.0, 85.3, 83.1, 88.0, 89.7]
    info = [92.4, 90.9, 90.9, 97.1, 84.6, 79.9]
    ti = [75.2, 76.2, 76.5, 80.4, 72.5, 69.6]

    ax.plot(scales, truth, "s--", color=C_TRUTH, linewidth=1.3,
            markersize=6, label="Truth %", alpha=0.85)
    ax.plot(scales, info, "^--", color=C_INFO, linewidth=1.3,
            markersize=6, label="Info %", alpha=0.85)
    ax.plot(scales, ti, "o-", color=C_TI, linewidth=2.0,
            markersize=7, label=r"T$\times$I %", zorder=5)

    ax.set_xlabel(r"Steering scale $\alpha$")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(65, 102)
    ax.legend(loc="lower left", frameon=True, fancybox=False, edgecolor="0.5")

    fig.tight_layout()
    save_both(fig, "scale_sweep")
    plt.close(fig)


# ---------------------------------------------------------------
# 4. LoRA COMBINATION
# ---------------------------------------------------------------
def plot_lora_combo():
    fig, ax = plt.subplots(figsize=(COLUMN_W, 2.0))

    methods = ["Baseline", "LoRA\n(DPO)", "MAST", "LoRA\n+ MAST"]
    truth = [64.1, 81.6, 83.1, 82.1]
    info = [85.1, 93.8, 97.1, 97.1]
    ti = [54.6, 76.5, 80.4, 79.7]

    x = np.arange(len(methods))
    w = 0.26

    ax.bar(x - w, truth, w, color=C_TRUTH, edgecolor="black",
           linewidth=0.5, label="Truth %", alpha=0.85)
    ax.bar(x, info, w, color=C_INFO, edgecolor="black",
           linewidth=0.5, label="Info %", alpha=0.85, hatch="///")
    ax.bar(x + w, ti, w, color=C_TI, edgecolor="black",
           linewidth=0.5, label=r"T$\times$I %")

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", ncol=3, frameon=True,
              fancybox=False, edgecolor="0.5", columnspacing=0.8)
    ax.grid(axis="y")

    fig.tight_layout()
    save_both(fig, "lora_combo")
    plt.close(fig)


# ---------------------------------------------------------------
# 5. COHERENCE DEGRADATION
# ---------------------------------------------------------------
def plot_degradation():
    fig, ax = plt.subplots(figsize=(COLUMN_W, 2.0))

    # These are from the existing bn=16 + LoRA data; bn=8 pending re-run.
    tasks = ["ARC-Easy", "ARC-Chal.", "HellaSwag", "MMLU"]
    standalone = [-7.8, -4.4, -2.2, -4.3]
    with_lora = [-3.6, -1.1, -0.3, -1.2]

    x = np.arange(len(tasks))
    w = 0.38

    ax.bar(x - w / 2, standalone, w, color=C_BASELINE, edgecolor="black",
           linewidth=0.5, label="MAST", hatch="\\\\\\")
    ax.bar(x + w / 2, with_lora, w, color=C_OURS, edgecolor="black",
           linewidth=0.5, label="MAST + LoRA-DPO")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylabel("Δ from baseline (pp)")
    ax.set_ylim(-10, 2)
    ax.legend(loc="lower right", frameon=True,
              fancybox=False, edgecolor="0.5")

    # Annotate values
    for i, (s, l) in enumerate(zip(standalone, with_lora)):
        ax.text(i - w / 2, s - 0.5, f"{s:.1f}", ha="center",
                va="top", fontsize=7)
        ax.text(i + w / 2, l - 0.5, f"{l:.1f}", ha="center",
                va="top", fontsize=7)

    fig.tight_layout()
    save_both(fig, "degradation")
    plt.close(fig)


# ---------------------------------------------------------------
# 6. TRAINING SIZE
# ---------------------------------------------------------------
def plot_training_size():
    # Fixed-split pipeline (test=408), seed=42. train=309 point reuses the
    # seed=42 run from the main 10-seed sweep, which also used the fixed split.
    fig, ax = plt.subplots(figsize=(COLUMN_W, 2.0))

    sizes = [50, 100, 200, 309]
    truth = [68.9, 77.5, 84.6, 83.6]
    info  = [96.6, 93.4, 90.7, 97.1]
    ti    = [66.2, 70.8, 75.2, 80.9]

    ax.plot(sizes, truth, "s--", color=C_TRUTH, linewidth=1.3,
            markersize=6, label="Truth %", alpha=0.85)
    ax.plot(sizes, info, "^--", color=C_INFO, linewidth=1.3,
            markersize=6, label="Info %", alpha=0.85)
    ax.plot(sizes, ti, "o-", color=C_TI, linewidth=2.0,
            markersize=7, label=r"T$\times$I %", zorder=5)

    ax.set_xlabel("Training set size")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(60, 102)
    ax.legend(loc="lower right", frameon=True,
              fancybox=False, edgecolor="0.5")

    fig.tight_layout()
    save_both(fig, "training_size")
    plt.close(fig)


# ---------------------------------------------------------------
# 7. POOL SIZE SWEEP
# ---------------------------------------------------------------
def plot_pool_size():
    # Fixed-split pipeline (test=408), seed=42, train=309 fixed. Varies CAA
    # extraction pool size from 1 to 100.
    fig, ax = plt.subplots(figsize=(COLUMN_W, 2.0))

    pools = [1, 5, 10, 50, 100]
    truth = [96.3, 89.2, 84.3, 80.6, 83.6]
    info  = [24.5, 84.1, 89.5, 95.3, 97.1]
    ti    = [22.5, 73.3, 74.0, 76.2, 80.9]

    ax.plot(pools, truth, "s--", color=C_TRUTH, linewidth=1.3,
            markersize=6, label="Truth %", alpha=0.85)
    ax.plot(pools, info, "^--", color=C_INFO, linewidth=1.3,
            markersize=6, label="Info %", alpha=0.85)
    ax.plot(pools, ti, "o-", color=C_TI, linewidth=2.0,
            markersize=7, label=r"T$\times$I %", zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel("CAA extraction pool size")
    ax.set_ylabel("Score (%)")
    ax.set_xticks(pools)
    ax.set_xticklabels([str(p) for p in pools])
    ax.set_ylim(15, 105)
    ax.legend(loc="lower right", frameon=True,
              fancybox=False, edgecolor="0.5")

    # Annotate pool=1 collapse
    ax.annotate("mode collapse\n(over-refusal)",
                xy=(1, 22.5), xytext=(2.5, 35),
                fontsize=7, color="0.3",
                arrowprops=dict(arrowstyle="->", color="0.5", lw=0.6))

    fig.tight_layout()
    save_both(fig, "pool_size")
    plt.close(fig)


if __name__ == "__main__":
    print(f"Generating figures in {OUTDIR}...")
    plot_noise_ablation()
    plot_bottleneck()
    plot_scale_sweep()
    plot_lora_combo()
    plot_degradation()
    plot_training_size()
    plot_pool_size()
    print("Done.")
