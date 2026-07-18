#!/usr/bin/env python3
"""Generate all figures and tables for the meeting presentation."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT_DIR = "paper/figures"

# ── Shared style ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 180,
})

COLORS = {
    "CD": "#9E9E9E",
    "RepEdit": "#42A5F5",
    "PEFT": "#FF7043",
    "Ours": "#66BB6A",
    "Combo": "#AB47BC",
}


# ══════════════════════════════════════════════════════════════════
# 1. MAIN COMPARISON TABLE (T*I ranking)
# ══════════════════════════════════════════════════════════════════
def fig1_main_comparison():
    """Bar chart: T*I for all methods including ours."""
    methods = [
        ("SH2",            "CD",      42.2),
        ("ITI",            "RepEdit", 54.9),
        ("TrFr",           "RepEdit", 54.6),
        ("LoRA (SFT)",     "PEFT",    58.3),
        ("TruthX",         "RepEdit", 62.8),
        ("LoRA DPO only",  "PEFT",    65.7),
        ("RED",            "PEFT",    70.8),
        ("AdaLoRA",        "PEFT",    70.1),
        ("Sora",           "PEFT",    76.0),
        ("Ours (standalone)", "Ours", 80.6),
        ("LoRA (DPO)",     "PEFT",    76.5),
        ("RaLFiT",         "PEFT",    77.4),
        ("Ours + LoRA DPO", "Combo",  79.7),
    ]
    methods.sort(key=lambda x: x[2])

    names = [m[0] for m in methods]
    tis = [m[2] for m in methods]
    cats = [m[1] for m in methods]
    colors = [COLORS[c] for c in cats]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(names)), tis, color=colors, edgecolor="white",
                   linewidth=0.5, height=0.7)

    for i, (n, c, ti) in enumerate(methods):
        if c in ("Ours", "Combo"):
            bars[i].set_edgecolor("#2E7D32" if c == "Ours" else "#7B1FA2")
            bars[i].set_linewidth(2.5)
        ax.text(ti + 0.4, i, f"{ti:.1f}%", va="center", fontsize=9,
                fontweight="bold" if c in ("Ours", "Combo") else "normal")

    ax.axvline(54.56, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(54.56 + 0.3, len(methods) - 0.3, "baseline ≈55%",
            fontsize=8, color="gray", va="top")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Truth * Informativeness (%)")
    ax.set_title("TruthfulQA T*I — Llama-2-7B-chat", fontweight="bold")
    ax.set_xlim(35, 88)
    ax.grid(axis="x", alpha=0.15)

    handles = [mpatches.Patch(color=COLORS[k], label=l) for k, l in [
        ("CD", "Contrastive Decoding"), ("RepEdit", "Representation Editing"),
        ("PEFT", "Fine-tuning (PEFT)"), ("Ours", "Ours (Activation Steering)"),
        ("Combo", "Ours + LoRA DPO"),
    ]]
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/main_results.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_DIR}/main_results.png")


# ══════════════════════════════════════════════════════════════════
# 2. PERFORMANCE DEGRADATION TABLE
# ══════════════════════════════════════════════════════════════════
def fig2_degradation():
    """Grouped bar: coherence degradation across benchmarks."""
    tasks = ["ARC-Easy", "ARC-Challenge", "HellaSwag", "MMLU"]
    baseline =      [68.5, 43.3, 76.0, 46.5]
    # LoRA+ours baseline is slightly different (LoRA changes the model)
    lora_baseline = [62.4, 40.9, 76.2, 46.1]
    ours =          [60.7, 38.9, 73.8, 42.2]
    lora_combo =    [58.8, 39.8, 75.9, 44.9]
    ralfit =        [None, None, 79.83, 46.77]

    deltas_ours = [o - b for o, b in zip(ours, baseline)]
    deltas_lora = [l - lb for l, lb in zip(lora_combo, lora_baseline)]
    deltas_ralfit = [r - b if r else None for r, b in zip(ralfit, baseline)]

    x = np.arange(len(tasks))
    w = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, deltas_ours, w, label="Ours standalone", color=COLORS["Ours"], alpha=0.85)
    ax.bar(x, deltas_lora, w, label="Ours + LoRA DPO", color=COLORS["Combo"], alpha=0.85)

    ralfit_x = [i for i, d in enumerate(deltas_ralfit) if d is not None]
    ralfit_d = [d for d in deltas_ralfit if d is not None]
    ax.bar([x[i] + w for i in ralfit_x], ralfit_d, w, label="RaLFiT", color=COLORS["PEFT"], alpha=0.85)

    for i, d in enumerate(deltas_ours):
        ax.text(x[i] - w, d - 0.5, f"{d:.1f}", ha="center", va="top", fontsize=8)
    for i, d in enumerate(deltas_lora):
        ax.text(x[i], d - 0.5, f"{d:.1f}", ha="center", va="top", fontsize=8)
    for i in ralfit_x:
        ax.text(x[i] + w, deltas_ralfit[i] - 0.5, f"{deltas_ralfit[i]:.1f}", ha="center", va="top", fontsize=8)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=10)
    ax.set_ylabel("Delta from Baseline (pp)")
    ax.set_title("Coherence Degradation on General Benchmarks", fontweight="bold")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(axis="y", alpha=0.15)
    ax.set_ylim(-12, 2)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/comparison_degradation.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_DIR}/comparison_degradation.png")


# ══════════════════════════════════════════════════════════════════
# 3. PERFORMANCE vs SCALE + REFUSAL RATE
# ══════════════════════════════════════════════════════════════════
def fig3_scale_sweep():
    """Line plot: Truth, Info, T*I, and refusal rate vs steering scale."""
    scales = [0.80, 0.90, 0.95, 1.00, 1.10, 1.20]
    truth  = [82.6, 85.0, 85.3, 86.0, 88.0, 89.7]
    info   = [92.4, 90.9, 90.9, 90.2, 84.6, 79.9]
    ti     = [75.2, 76.2, 76.5, 76.5, 72.5, 69.6]
    refusal= [11.0, 13.7, 15.2, 17.2, 23.5, 29.4]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 1]})

    # Left panel: T*I, Truth, Info
    ax1.plot(scales, ti, "o-", color="#2E7D32", linewidth=2.5, markersize=8, label="T*I", zorder=10)
    ax1.plot(scales, truth, "s--", color="#1565C0", linewidth=1.5, markersize=6, label="Truth", alpha=0.8)
    ax1.plot(scales, info, "^--", color="#E65100", linewidth=1.5, markersize=6, label="Info", alpha=0.8)
    ax1.axhline(77.4, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax1.text(1.15, 78.0, "RaLFiT", fontsize=8, color="gray")
    ax1.set_xlabel("Steering Scale")
    ax1.set_ylabel("Score (%)")
    ax1.set_ylim(65, 95)
    ax1.set_title("Performance vs Scale", fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.15)

    # Right panel: Refusal rate
    ax2.fill_between(scales, refusal, alpha=0.2, color="red")
    ax2.plot(scales, refusal, "o-", color="red", linewidth=2, markersize=7)
    for s, r in zip(scales, refusal):
        ax2.text(s, r + 1, f"{r:.0f}%", ha="center", fontsize=9, color="red")
    ax2.set_xlabel("Steering Scale")
    ax2.set_ylabel("Refusal Rate (%)")
    ax2.set_title("Model Refusal Rate vs Scale", fontweight="bold")
    ax2.set_ylim(0, 38)
    ax2.grid(alpha=0.15)

    fig.suptitle("Steering Scale Controls Truth-Informativeness Tradeoff via Refusal Behavior",
                  fontweight="bold", fontsize=13, y=1.02)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/scaling_trends.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_DIR}/scaling_trends.png")


# ══════════════════════════════════════════════════════════════════
# 4. PERFORMANCE vs SAMPLE SIZE
# ══════════════════════════════════════════════════════════════════
def fig4_pool_size():
    """Line plot: T*I vs number of steering pool examples."""
    pools = [10, 50, 100]
    ti    = [71.1, 73.1, 76.5]
    mc    = [53.0, 53.9, 56.9]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pools, ti, "o-", color="#2E7D32", linewidth=2.5, markersize=10,
            label="T*I (GPT Judge)")
    ax.plot(pools, mc, "s--", color="#42A5F5", linewidth=1.5, markersize=7,
            label="MC Accuracy", alpha=0.8)

    for i, (p, t) in enumerate(zip(pools, ti)):
        ax.annotate(f"{t:.1f}%", (p, t), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=10, fontweight="bold",
                    color="#2E7D32")

    ax.axhline(56.2, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.text(15, 56.8, "Baseline T*I (no steering)", fontsize=8, color="gray")

    ax.set_xlabel("Steering Pool Size (number of examples)")
    ax.set_ylabel("Score (%)")
    ax.set_title("Data Efficiency: T*I vs Extraction Pool Size", fontweight="bold")
    ax.set_xticks(pools)
    ax.set_xlim(5, 110)
    ax.set_ylim(50, 82)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.15)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/line_pool_size.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_DIR}/line_pool_size.png")


# ══════════════════════════════════════════════════════════════════
# 5. KL DIVERGENCE TRADEOFF
# ══════════════════════════════════════════════════════════════════
def fig5_kl_effect():
    """KL weight vs T*I and coherence degradation side by side."""
    kl_weights = ["0\n(no KL)", "0.01", "0.1"]
    ti_scores = [76.5, 75.0, 72.8]
    mmlu_delta = [-4.4, -6.1, -2.6]
    hellaswag_delta = [-2.2, -3.1, -0.9]

    x = np.arange(len(kl_weights))
    w = 0.3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bars1 = ax1.bar(x, ti_scores, w*2, color=["#66BB6A", "#81C784", "#A5D6A7"], edgecolor="white")
    for i, v in enumerate(ti_scores):
        ax1.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=11)
    ax1.axhline(77.4, color="gray", linestyle=":", alpha=0.5)
    ax1.text(2.4, 77.8, "RaLFiT", fontsize=8, color="gray")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"KL = {w}" for w in kl_weights])
    ax1.set_ylabel("T*I (%)")
    ax1.set_title("TruthfulQA Performance", fontweight="bold")
    ax1.set_ylim(68, 82)
    ax1.grid(axis="y", alpha=0.15)

    ax2.bar(x - w/2, mmlu_delta, w, label="MMLU", color="#1565C0", alpha=0.85)
    ax2.bar(x + w/2, hellaswag_delta, w, label="HellaSwag", color="#42A5F5", alpha=0.85)
    for i, (m, h) in enumerate(zip(mmlu_delta, hellaswag_delta)):
        ax2.text(x[i] - w/2, m - 0.4, f"{m:.1f}", ha="center", va="top", fontsize=8)
        ax2.text(x[i] + w/2, h - 0.4, f"{h:.1f}", ha="center", va="top", fontsize=8)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.axhline(-0.5, color="gray", linestyle=":", alpha=0.5)
    ax2.text(2.4, -0.1, "RaLFiT MMLU", fontsize=8, color="gray")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"KL = {w}" for w in kl_weights])
    ax2.set_ylabel("Delta from Baseline (pp)")
    ax2.set_title("Coherence Degradation", fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.set_ylim(-8, 1)
    ax2.grid(axis="y", alpha=0.15)

    fig.suptitle("KL Divergence Regularization: Truthfulness vs Coherence Tradeoff",
                 fontweight="bold", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/kl_tradeoff.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_DIR}/kl_tradeoff.png")


# ══════════════════════════════════════════════════════════════════
# 6. NOISE ABLATION
# ══════════════════════════════════════════════════════════════════
def fig6_noise_ablation():
    """Bar chart: real vectors vs noise ablation."""
    methods = ["Baseline\n(no steering)", "Noise Vectors\n(ablation)", "CAA Vectors\n(ours)"]
    ti = [56.2, 60.8, 76.5]
    colors = ["#9E9E9E", "#FFAB91", "#66BB6A"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(methods)), ti, color=colors, edgecolor="white",
                  linewidth=1, width=0.5)
    bars[-1].set_edgecolor("#2E7D32")
    bars[-1].set_linewidth(2.5)

    for i, v in enumerate(ti):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=12)

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel("T*I (%)")
    ax.set_title("Noise Ablation: Steering Vectors Carry Real Signal", fontweight="bold")
    ax.set_ylim(0, 88)
    ax.grid(axis="y", alpha=0.15)

    pass  # Clean chart, annotations removed

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/noise_ablation.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_DIR}/noise_ablation.png")


# ══════════════════════════════════════════════════════════════════
# 7. COMBINATION: LoRA DPO + OURS
# ══════════════════════════════════════════════════════════════════
def fig7_combination():
    """Bar chart showing complementary contributions."""
    methods = ["Baseline", "LoRA DPO\nonly", "Ours\nstandalone", "LoRA DPO\n+ Ours"]
    ti = [56.2, 65.7, 76.5, 79.7]
    truth = [59.0, 68.6, 86.0, 82.1]
    info = [96.4, 97.1, 90.2, 97.1]

    colors = ["#9E9E9E", "#FF7043", "#66BB6A", "#AB47BC"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(range(len(methods)), ti, color=colors, edgecolor="white",
                  linewidth=1, width=0.6)
    bars[-1].set_edgecolor("#7B1FA2")
    bars[-1].set_linewidth(2.5)

    for i, (t, tr, inf) in enumerate(zip(ti, truth, info)):
        ax.text(i, t + 1.5, f"T*I = {t:.1f}%", ha="center", fontweight="bold", fontsize=10)
        ax.text(i, t - 4, f"T={tr:.0f}% I={inf:.0f}%", ha="center", fontsize=8,
                color="white" if t > 60 else "black")

    ax.axhline(77.4, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.text(3.4, 77.8, "RaLFiT SOTA", fontsize=8, color="gray")

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel("T*I (%)")
    ax.set_title("Complementary Methods: Parameter Space (LoRA) + Activation Space (Ours)",
                 fontweight="bold")
    ax.set_ylim(0, 90)
    ax.grid(axis="y", alpha=0.15)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/combination_lora.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_DIR}/combination_lora.png")


# ══════════════════════════════════════════════════════════════════
# 8. PARAMS vs PERFORMANCE (scatter)
# ══════════════════════════════════════════════════════════════════
def fig8_params_scatter():
    """Scatter: trainable params vs T*I."""
    est = {
        "ITI":              (131,    54.9, "RepEdit"),
        "TruthX":           (315000, 62.8, "RepEdit"),
        "RED":              (4200,   70.8, "PEFT"),
        "LoRA (SFT)":       (5960,   58.3, "PEFT"),
        "LoRA (DPO)":       (5960,   76.5, "PEFT"),
        "AdaLoRA":          (5960,   70.1, "PEFT"),
        "Sora":             (4200,   76.0, "PEFT"),
        "RaLFiT":           (5110,   77.4, "PEFT"),
        "Ours":             (66,     80.6, "Ours"),
        "Ours+LoRA":        (5176,   79.7, "Combo"),
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, (pk, ti, cat) in est.items():
        c = COLORS[cat]
        sz = 140 if cat in ("Ours", "Combo") else 80
        ec = "#2E7D32" if cat == "Ours" else "#7B1FA2" if cat == "Combo" else "white"
        lw = 2.5 if cat in ("Ours", "Combo") else 0.5
        z = 10 if cat in ("Ours", "Combo") else 5
        ax.scatter(pk, ti, s=sz, c=c, edgecolors=ec, linewidths=lw, zorder=z, alpha=0.85)

        if name == "Ours":
            ax.annotate(f"{name}\n66K params", (pk, ti), xytext=(-10, -22),
                        textcoords="offset points", fontsize=9, fontweight="bold",
                        color="#2E7D32", ha="center",
                        arrowprops=dict(arrowstyle="-", color="#2E7D32", lw=0.8))
        elif name == "Ours+LoRA":
            ax.annotate(f"{name}\n79.7% T*I", (pk, ti), xytext=(10, 10),
                        textcoords="offset points", fontsize=9, fontweight="bold",
                        color="#7B1FA2", ha="left",
                        arrowprops=dict(arrowstyle="-", color="#7B1FA2", lw=0.8))
        elif name == "TruthX":
            ax.annotate(name, (pk, ti), xytext=(5, -12),
                        textcoords="offset points", fontsize=8, color="gray")
        else:
            ax.annotate(name, (pk, ti), xytext=(5, 5),
                        textcoords="offset points", fontsize=8, color="gray")

    ax.set_xscale("log")
    ax.set_xlabel("Trainable Parameters (log scale)")
    ax.set_ylabel("T*I (%)")
    ax.set_title("Performance vs Compute — TruthfulQA Llama-2-7B-chat", fontweight="bold")
    ax.set_xlim(50, 800000)
    ax.set_ylim(50, 84)
    ax.set_xticks([100, 1000, 10000, 100000])
    ax.set_xticklabels(["100K", "1M", "10M", "100M"])
    ax.grid(True, alpha=0.15)

    ax.axhline(54.56, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.text(60, 55.0, "baseline", fontsize=8, color="gray")

    handles = [mpatches.Patch(color=COLORS[k], label=l) for k, l in [
        ("RepEdit", "Representation Editing"), ("PEFT", "Fine-tuning (PEFT)"),
        ("Ours", "Ours (66K params)"), ("Combo", "Ours + LoRA DPO"),
    ]]
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/comparison_compute.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_DIR}/comparison_compute.png")


if __name__ == "__main__":
    fig1_main_comparison()
    fig2_degradation()
    fig3_scale_sweep()
    fig4_pool_size()
    fig5_kl_effect()
    fig6_noise_ablation()
    fig7_combination()
    fig8_params_scatter()
    print(f"\nDone! All 8 figures in {OUT_DIR}/")
