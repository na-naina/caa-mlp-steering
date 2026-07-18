"""Produce the design-history figures: timeline and architecture evolution."""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def timeline_figure(out_path: Path) -> None:
    milestones = [
        ("SAE feature\nprobing (diagnostic)", "Jan 2025", 0, False),
        ("MLP-on-vectors\nidea (prof)", "Sep 2025", 1, False),
        ("Fat MLP era\n134M params,\nno residual", "Sep 25 - Feb 26", 2, False),
        ("Breakthrough\nresidual +\nbottleneck k=16", "Feb 16, 2026", 3, True),
        ("Bottleneck sweep:\nk=8 picked", "Mar 23, 2026", 4, False),
        ("10-seed\nmain result", "Apr 2026", 5, False),
    ]

    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-1.2, 1.5)
    ax.axis("off")

    ax.plot([-0.3, 5.3], [0, 0], color="gray", lw=1.2, zorder=1)
    for _, _, xpos, _ in milestones:
        ax.plot([xpos], [0], "|", color="gray", markersize=10, zorder=2)

    for label, date, xpos, is_breakthrough in milestones:
        fc = "#d1e3ff" if is_breakthrough else "#f0f0f0"
        ec = "#2b5cb3" if is_breakthrough else "#888888"
        lw = 2.0 if is_breakthrough else 1.0
        box = FancyBboxPatch(
            (xpos - 0.42, 0.18), 0.84, 0.88,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3,
        )
        ax.add_patch(box)
        weight = "bold" if is_breakthrough else "normal"
        ax.text(xpos, 0.62, label, ha="center", va="center", fontsize=8.5, fontweight=weight, zorder=4)
        ax.text(xpos, -0.18, date, ha="center", va="top", fontsize=7.5, style="italic", color="#555555")

    for i in range(len(milestones) - 1):
        ax.annotate("",
                    xy=(milestones[i + 1][2] - 0.43, 0.62),
                    xytext=(milestones[i][2] + 0.43, 0.62),
                    arrowprops=dict(arrowstyle="->", color="#888888", lw=1.0))

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"saved {out_path.with_suffix('.pdf')}")


def arch_figure(out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 4.6))

    # ----- Before -----
    ax = axes[0]
    ax.set_xlim(0, 12)
    ax.set_ylim(-1.2, 1.4)
    ax.axis("off")
    ax.set_title("Before (pre-Feb 16, 2026): fat non-residual MLP, ~134M params at d=4096",
                 fontsize=11, fontweight="bold", loc="left")

    blocks = [
        ("$v_{CAA}$", 0.5, "#c6e8c6", "#388e3c"),
        ("Linear\n$d \\to 2d$", 3.0, "#f5c6c6", "#c62828"),
        ("Linear\n$2d \\to 2d$", 5.5, "#f5c6c6", "#c62828"),
        ("Linear\n$2d \\to d$", 8.0, "#f5c6c6", "#c62828"),
        ("$v$", 10.5, "#c6e8c6", "#388e3c"),
    ]
    for label, x, fc, ec in blocks:
        box = FancyBboxPatch((x - 0.7, -0.35), 1.4, 0.7,
                             boxstyle="round,pad=0.03,rounding_size=0.08",
                             facecolor=fc, edgecolor=ec, linewidth=1.3)
        ax.add_patch(box)
        ax.text(x, 0.0, label, ha="center", va="center", fontsize=10)
    for i in range(len(blocks) - 1):
        ax.annotate("", xy=(blocks[i + 1][1] - 0.75, 0), xytext=(blocks[i][1] + 0.75, 0),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

    # ----- After -----
    ax = axes[1]
    ax.set_xlim(0, 12)
    ax.set_ylim(-1.5, 1.4)
    ax.axis("off")
    ax.set_title("After (Feb 16, 2026): residual bottleneck MLP, 66K params at k=8",
                 fontsize=11, fontweight="bold", loc="left")

    blocks2 = [
        ("$v_{CAA}$", 0.5, "#c6e8c6", "#388e3c"),
        ("Linear\n$d \\to k$", 3.0, "#c6d8f5", "#1565c0"),
        ("ReLU", 5.0, "#c6d8f5", "#1565c0"),
        ("Linear\n$k \\to d$", 7.0, "#c6d8f5", "#1565c0"),
        ("$+$", 9.0, "#fff0c6", "#f57c00"),
        ("$v$", 10.8, "#c6e8c6", "#388e3c"),
    ]
    widths = [1.4, 1.4, 1.0, 1.4, 0.6, 1.4]
    for (label, x, fc, ec), w in zip(blocks2, widths):
        box = FancyBboxPatch((x - w / 2, -0.35), w, 0.7,
                             boxstyle="round,pad=0.03,rounding_size=0.08",
                             facecolor=fc, edgecolor=ec, linewidth=1.3)
        ax.add_patch(box)
        ax.text(x, 0.0, label, ha="center", va="center", fontsize=10)

    for i in range(len(blocks2) - 1):
        ax.annotate("",
                    xy=(blocks2[i + 1][1] - widths[i + 1] / 2 - 0.05, 0),
                    xytext=(blocks2[i][1] + widths[i] / 2 + 0.05, 0),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

    # Residual skip arrow
    ax.annotate("",
                xy=(blocks2[4][1], -0.4),
                xytext=(blocks2[0][1], -0.4),
                arrowprops=dict(arrowstyle="-", color="#f57c00", lw=1.8, connectionstyle="arc3,rad=-0.3"))
    ax.annotate("",
                xy=(blocks2[4][1], -0.4),
                xytext=(blocks2[4][1] - 0.01, -0.4 + 0.01),
                arrowprops=dict(arrowstyle="->", color="#f57c00", lw=1.8))
    ax.text((blocks2[0][1] + blocks2[4][1]) / 2, -1.05, "residual skip", ha="center",
            fontsize=9, color="#f57c00", style="italic")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"saved {out_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    out_dir = Path("paper/figures/paper")
    timeline_figure(out_dir / "design_timeline")
    arch_figure(out_dir / "arch_evolution")
