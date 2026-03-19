#!/usr/bin/env python3
"""Generate comparison figures for meeting presentation."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR = "paper/figures"

# ── Data from RaLFiT table + ours ──────────────────────────────────
# RaLFiT paper baseline: T=64.14, I=85.07, TI=54.56
RB_TI = 54.56
# Our baseline: T=59.0, I=96.4, TI=56.2
OB_TI = 56.2

methods = [
    # name,           category,    T*I
    ("ITI",           "RepEdit",   54.9),
    ("TrFr",          "RepEdit",   54.6),
    ("TruthX",        "RepEdit",   62.8),
    ("SH2",           "CD",        42.2),
    ("RED",           "PEFT",      70.8),
    ("LoRA (SFT)",    "PEFT",      58.3),
    ("LoRA (DPO)",    "PEFT",      76.5),
    ("AdaLoRA",       "PEFT",      70.1),
    ("Sora",          "PEFT",      76.0),
    ("RaLFiT",        "PEFT",      77.4),
    ("Ours (bn=16)",  "Ours",      76.5),
]

CAT_COLORS = {
    "CD":      "#9E9E9E",
    "RepEdit": "#42A5F5",
    "PEFT":    "#FF7043",
    "Ours":    "#66BB6A",
}
CAT_LABELS = {
    "CD":      "Contrastive Decoding",
    "RepEdit": "Representation Editing",
    "PEFT":    "Fine-tuning (PEFT)",
    "Ours":    "Ours (Inference-time MLP)",
}


def fig1_ti_comparison():
    """Bar chart: T*I absolute for all methods."""
    data = [(n, c, ti) for n, c, ti in methods]
    data.sort(key=lambda x: x[2])

    names = [d[0] for d in data]
    tis = [d[2] for d in data]
    cats = [d[1] for d in data]
    colors = [CAT_COLORS[c] for c in cats]

    fig, ax = plt.subplots(figsize=(12, 5.5))

    bars = ax.barh(range(len(names)), tis, color=colors, edgecolor="white", linewidth=0.5, height=0.7)

    for i, (n, c, ti) in enumerate(data):
        if c == "Ours":
            bars[i].set_edgecolor("#2E7D32")
            bars[i].set_linewidth(2.5)

    for i, (n, c, ti) in enumerate(data):
        ax.text(ti + 0.5, i, f"{ti:.1f}%",
                va="center", fontsize=9, fontweight="bold" if c == "Ours" else "normal")

    ax.axvline(RB_TI, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(RB_TI + 0.3, len(data) - 0.3, f"baseline ≈{RB_TI:.0f}%",
            fontsize=8, color="gray", va="top")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Truth × Informativeness (%)", fontsize=11)
    ax.set_title("TruthfulQA T*I — Llama-2-7B-chat", fontsize=13, fontweight="bold")
    ax.set_xlim(35, 90)
    ax.grid(axis="x", alpha=0.15)

    handles = [mpatches.Patch(color=CAT_COLORS[k], label=CAT_LABELS[k])
               for k in ["CD", "RepEdit", "PEFT", "Ours"]]
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/comparison_ti_bars.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_DIR}/comparison_ti_bars.png")


def fig2_compute_vs_performance():
    """Scatter: params (K) vs T*I. Params estimated from architectures."""
    fig, ax = plt.subplots(figsize=(9, 6))

    # params_k = trainable parameters in thousands
    # Sources:
    #   ITI: 32 layers × 32 heads × 128-dim probes ≈ 131K (Li et al. NeurIPS 2023)
    #   TruthX: 3 MLPs (2 encoders + decoder) × 10 layers ≈ 315M (Zhang et al. ACL 2024)
    #   LoRA (SFT): 5.96M — RaLFiT paper Table 2 (Li et al. ACL 2025)
    #   LoRA (DPO): 5.96M — RaLFiT paper Table 2
    #   AdaLoRA: 5.96M — same LoRA arch, adaptive rank (RaLFiT Table 2)
    #   RED/Sora: LoRA-based, ~4-6M range (no exact count in paper)
    #   RaLFiT: 5.11M — RaLFiT paper Table 2 (rank-adaptive, avg budget 8, W^O+W^D)
    #   Ours: linear(4096,16) + linear(16,4096) = 131K + bias = ~135K
    est = {
        # name:          (params_K,  T*I,  category)
        "ITI":           (131,       54.9, "RepEdit"),
        "TruthX":        (315000,    62.8, "RepEdit"),
        "RED":           (4200,      70.8, "PEFT"),
        "LoRA (SFT)":    (5960,      58.3, "PEFT"),
        "LoRA (DPO)":    (5960,      76.5, "PEFT"),
        "AdaLoRA":       (5960,      70.1, "PEFT"),
        "Sora":          (4200,      76.0, "PEFT"),
        "RaLFiT":        (5110,      77.4, "PEFT"),
        "Ours (bn=16)":  (135,       76.5, "Ours"),
    }

    for name, (params_k, ti, cat) in est.items():
        size = 120 if cat == "Ours" else 80
        color = CAT_COLORS[cat]
        zorder = 10 if cat == "Ours" else 5
        edgecolor = "#2E7D32" if cat == "Ours" else "white"
        lw = 2.5 if cat == "Ours" else 0.5

        ax.scatter(params_k, ti, s=size, c=color, edgecolors=edgecolor,
                   linewidths=lw, zorder=zorder, alpha=0.85)

        if name == "Ours (bn=16)":
            ax.annotate(f"{name}\n135K params", (params_k, ti), xytext=(-10, -22),
                        textcoords="offset points", fontsize=9, fontweight="bold",
                        color="#2E7D32", ha="center",
                        arrowprops=dict(arrowstyle="-", color="#2E7D32", lw=0.8))
        elif name == "TruthX":
            ax.annotate(name, (params_k, ti), xytext=(5, -12),
                        textcoords="offset points", fontsize=8, ha="left", color="gray")
        else:
            ax.annotate(name, (params_k, ti), xytext=(5, 5),
                        textcoords="offset points", fontsize=8, ha="left", color="gray")

    ax.set_xscale("log")
    ax.set_xlabel("Trainable Parameters (log scale)", fontsize=11)
    ax.set_ylabel("T*I (%)", fontsize=11)
    ax.set_title("Performance vs Compute — TruthfulQA Llama-2-7B-chat",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(50, 800000)
    ax.set_ylim(50, 82)
    ax.grid(True, alpha=0.15)

    # Custom x-axis labels
    ax.set_xticks([100, 1000, 10000, 100000])
    ax.set_xticklabels(["100K", "1M", "10M", "100M"])

    ax.axhline(RB_TI, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.text(60, RB_TI + 0.5, "baseline", fontsize=8, color="gray")

    handles = [mpatches.Patch(color=CAT_COLORS[k], label=CAT_LABELS[k])
               for k in ["RepEdit", "PEFT", "Ours"]]
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/comparison_compute.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_DIR}/comparison_compute.png")


if __name__ == "__main__":
    fig1_ti_comparison()
    fig2_compute_vs_performance()
    print("\nDone. All figures in paper/figures/")
