#!/usr/bin/env python3
"""Meeting deck organised by the 22-Sep revision checklist: one slide per item,
one figure/table each, a one-line status, minimal interpretation. Exploratory
interpretability results go to a labelled appendix.

Usage: .venv/bin/python scripts/make_meeting_deck2.py
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

O = Path("data/outputs")
OUT = Path("paper/figures/oct5_meeting")
TABLE1 = OUT / "_table1.png"
for f in OUT.glob("*.png"):
    if not f.name.startswith("_"):
        f.unlink()

C_MAST, C_DV, C_LORA, C_BASE, C_CAA = "#2a78d6", "#eb6834", "#1baf7a", "#8c8b86", "#b9b8b2"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#ececea", "axes.axisbelow": True,
                     "figure.facecolor": "white", "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
SLIDES = []
R = json.loads(Path("paper/figures/revision_oct2026/revision_results_gpt.json").read_text())
T = R["table"]
CELLS = [(s, f) for s in (42, 123, 456) for f in (1, 2)]


def m(k, metric="ti_product"):
    return T[k][metric]["mean"], T[k][metric]["sd"]


def new(title, status, ncols=1, **kw):
    fig, ax = plt.subplots(1, ncols, figsize=(13.33, 7.5), **kw)
    fig.text(0.04, 0.95, title, fontsize=21, fontweight="bold", color=INK, va="top")
    if status:
        import textwrap
        fig.text(0.04, 0.025, textwrap.fill(status, 165), fontsize=11.5, color=INK2, va="bottom")
    return fig, ax


def done(fig, name, **kw):
    fig.subplots_adjust(top=kw.get("top", 0.83), bottom=kw.get("bottom", 0.14), left=kw.get("left", 0.08),
                        right=kw.get("right", 0.97), wspace=kw.get("wspace", 0.3))
    fig.savefig(OUT / f"{len(SLIDES) + 1:02d}_{name}.png", dpi=110)
    SLIDES.append(fig)


def table(ax, header, rows, widths, fs=12, rowh=0.085, header_color="#f2f1ed"):
    ax.axis("off")
    x0 = 0
    xs = np.cumsum([0] + widths[:-1]) / sum(widths)
    y = 0.95
    for x, h in zip(xs, header):
        ax.text(x, y, h, fontsize=fs, fontweight="bold", color=INK, va="top", transform=ax.transAxes)
    ax.plot([0, 1], [y - 0.06, y - 0.06], color=INK2, lw=1, transform=ax.transAxes)
    y -= 0.09
    for r in rows:
        bold = r[0].startswith("**")
        for x, c in zip(xs, r):
            ax.text(x, y, c.strip("*"), fontsize=fs, color=INK, va="top", transform=ax.transAxes,
                    fontweight="bold" if bold else "normal")
        y -= rowh


def judged(p):
    d = json.loads(Path(p).read_text())
    return d["results"], d["stats"]


# ---------------------------------------------------------------- 0. status
fig, ax = new("Revision checklist (22 Sep) — status", "ARR resubmission due 12 Oct, same reviewers. Draft: paper/drafts/revision_oct2026/paper/main.tex")
table(ax, ["#", "Item", "Status", "Where"], [
    ["1", "Per-category breakdown (Rev. 74Zo)", "done", "App. C table + slide 2"],
    ["2", "Model setup aligned with the literature", "done", "§4 + slide 3"],
    ["3", "Direct-vector baseline across lrs; prior work; always vs sometimes", "done (LLaMA); Gemma running", "§5.2 + slides 4–5"],
    ["4", "Recent literature", "done, verified", "§2 + slide 6"],
    ["5", "Table 1 restructured + 'training' column", "done", "slide 7"],
    ["6", "T×I definition; alignment with original paper and RaLFiT", "done", "§4, App. B + slide 8"],
    ["7", "Both aggregations in appendix", "done", "App. B + slide 8"],
    ["8", "Multiple seeds for every method", "done (3 seeds × 2 folds)", "slide 9"],
    ["9", "Reviewer 3: add the already-run experiments", "done; 2 seed sets pending", "§5.4 + slide 10"],
    ["10", "Aligned with RaLFiT's protocol", "done", "§4 + slide 11"],
], widths=[0.4, 6.2, 2.6, 2.6], fs=13, rowh=0.083)
done(fig, "status", top=0.84)

# ---------------------------------------------------------------- 1. categories
cats = R["categories"]
cm = {c: v for c, v in cats["mast"].items() if v["n"] >= 30}
rows = sorted(((c, cats["baseline"][c]["ti_conj"], cm[c]["ti_conj"], cm[c]["n"] // 3) for c in cm), key=lambda t: t[2] - t[1])
fig, ax = new("1 · Per-category results (Reviewer 74Zo)",
              "Status: done — full 38-category table in Appendix C. Shown: categories with ≥10 questions; Truth∧Info, all 817 questions × 3 seeds.")
for y, (c, b0, b1, n) in enumerate(rows):
    ax.plot([b0, b1], [y, y], color="#d9d8d3", lw=3)
    ax.plot(b0, y, "o", color=C_BASE, ms=8, label="unsteered" if y == 0 else None)
    ax.plot(b1, y, "o", color=C_MAST, ms=9, label="MAST" if y == 0 else None)
ax.set_yticks(range(len(rows)), [f"{c} ({n})" for c, _, _, n in rows], fontsize=10)
ax.set_xlabel("Truth ∧ Info (%)")
ax.set_xlim(0, 100)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
done(fig, "categories", left=0.28, top=0.86)

# ---------------------------------------------------------------- 2. model alignment
fig, ax = new("2 · Model choice is the field's standard for this benchmark",
              "Status: done (§4). We add Gemma-3-4B-IT as a second family with the unchanged recipe (slide 10).")
table(ax, ["Paper", "Venue", "Model(s) for TruthfulQA", "Reports open-ended T×I?"], [
    ["ITI (Li et al.)", "NeurIPS 2023", "LLaMA-7B; LLaMA-2-7B-Chat in RaLFiT's re-run", "yes"],
    ["TruthX (Zhang et al.)", "ACL 2024", "LLaMA-2-7B-Chat (+ others)", "yes"],
    ["LoFiT (Yin et al.)", "NeurIPS 2024", "LLaMA-2-7B (base)", "partly (GPT-4 judge)"],
    ["BiPO (Cao et al.)", "NeurIPS 2024", "LLaMA-2-7B-Chat", "no (MC1/MC2 only)"],
    ["RaLFiT (Li et al.)", "ACL Findings 2025", "LLaMA-2-7B-Chat (+ LLaMA-3)", "yes"],
    ["IDEEA (Wang et al.)", "EMNLP Findings 2026", "LLaMA-2-7B (+ 5 others)", "yes (open judges)"],
    ["**Ours", "**", "**LLaMA-2-7B-Chat + Gemma-3-4B-IT", "**yes, both aggregations"],
], widths=[3, 2.4, 4.6, 3], fs=13, rowh=0.1)
done(fig, "models", top=0.84)

# ---------------------------------------------------------------- 3. direct vector (LLaMA)
fig, ax = new("3 · Direct vector vs MAST across learning rates (LLaMA-2-7B-Chat)",
              "Mean ± s.d. over 3 seeds × 2 folds. Same loss, data, steps; only the trainable object differs. "
              "Status: done. Answer to 'always or on occasion': on occasion (at MAST's lr), ties at each method's best lr.")
LR = {"5e-4": 5e-4, "1e-3": 1e-3, "2e-3": 2e-3, "5e-3": 5e-3}
for lab, pts, c, mk in (("MAST", {"5e-4": "mast", "1e-3": "mast_lr1e-3", "2e-3": "mast_lr2e-3"}, C_MAST, "o"),
                        ("direct vector (zero init)", {"5e-4": "dvzero_lr5e-4", "1e-3": "dvzero_lr1e-3", "2e-3": "dvzero_lr2e-3", "5e-3": "dvzero_lr5e-3"}, C_DV, "s")):
    ax.errorbar([LR[k] for k in pts], [m(v)[0] for v in pts.values()], yerr=[m(v)[1] for v in pts.values()],
                color=c, marker=mk, ms=9, lw=2.2, capsize=4, label=lab)
ax.axhline(m("loradpo")[0], color=C_LORA, lw=1.8, ls="--", label="LoRA-DPO (same data)")
ax.axhline(m("baseline")[0], color=C_BASE, lw=1.5, ls=":", label="unsteered")
ax.set_xscale("log")
ax.set_xticks(list(LR.values()), list(LR.keys()))
ax.minorticks_off()
ax.set_xlabel("learning rate")
ax.set_ylabel("True × Info (%)")
ax.set_ylim(50, 85)
ax.legend(frameon=False, loc="lower left", fontsize=12)
w = R["comparisons"]
ax.text(0.99, 0.04, "MAST ahead, of 6 cells\n"
        f"at 5e-4: {int(w['mast - dvzero_lr5e-4']['wins']['a_wins'])}/6\n"
        f"at 1e-3: {int(w['mast_lr1e-3 - dvzero_lr1e-3']['wins']['a_wins'])}/6\n"
        f"at 2e-3: {int(w['mast_lr2e-3 - dvzero_lr2e-3']['wins']['a_wins'])}/6\n"
        f"MAST@5e-4 vs vector@2e-3: {int(w['mast - dvzero_lr2e-3']['wins']['a_wins'])}/6",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=11.5, bbox=dict(fc="#f2f1ed", ec="none", boxstyle="round"))
done(fig, "direct_vector")

# ---------------------------------------------------------------- 4. direct vector (Gemma)
fig, ax = new("3b · Direct vector on Gemma-3-4B-IT (preliminary, 1 seed, fold 1)",
              "Status: running — lrs 3e-2 and 1e-1 pending. Bare-vector norm after training in brackets; MAST's learned correction has norm ≈ 180.")
pts = []
for lr in ("5e-4", "2e-3", "5e-3", "3e-2", "1e-1"):
    d = O / f"rg4b_dvzero_lr{lr}_s42/fold1"
    for jf in ("gpt_judge_results.json", "open_judge_results.json"):
        f = d / "mlp_mc/scale_1.00" / jf
        if f.exists():
            _, st = judged(f)
            pts.append((float(lr), 100 * st["truth_accuracy"] * st["info_accuracy"],
                        json.loads((d / "meta.json").read_text())["v_final_norm"]))
            break
_, gm = judged(O / "g4b_bn8_full/mlp_mc/scale_1.00/gpt_judge_results.json")
_, gb = judged(O / "g4b_bn8_full/baseline/scale_0.00/gpt_judge_results.json")
ax.plot([p[0] for p in pts], [p[1] for p in pts], color=C_DV, marker="s", ms=10, lw=2.2, label="direct vector (zero init)")
for x, y, n in pts:
    ax.text(x, y + 1.5, f"[{n:.0f}]", ha="center", fontsize=11, color=INK2)
ax.axhline(100 * gm["truth_accuracy"] * gm["info_accuracy"], color=C_MAST, lw=2.2, label="MAST, lr 5e-4 (unchanged from LLaMA)")
ax.axhline(100 * gb["truth_accuracy"] * gb["info_accuracy"], color=C_BASE, lw=1.5, ls=":", label="unsteered")
ax.set_xscale("log")
ax.set_xlim(3e-4, 2e-1)
ax.set_ylim(45, 95)
ax.set_xlabel("learning rate (direct vector)")
ax.set_ylabel("True × Info (%)")
ax.legend(frameon=False, loc="center left", fontsize=12)
done(fig, "direct_vector_gemma")

# ---------------------------------------------------------------- 5. literature
fig, ax = new("4 · Literature since submission (and direct-vector prior work)",
              "Status: done; key claims verified against the papers (docs/lit_update_oct2026.md).")
table(ax, ["Paper", "What it does", "Relation to us"], [
    ["BiPO (NeurIPS'24)", "single vector, preference loss, LLaMA-2-7B-Chat", "closest prior; MC only, one lr, no LoRA baseline"],
    ["LoFiT (NeurIPS'24)", "learned head offsets, DPO, 2-fold CV", "base model; lr sensitivity noted"],
    ["RED (ACL'24)", "learned vectors at every layer", "70.85 T×I in RaLFiT's table"],
    ["Dunefsky & Cohan (COLM'25)", "vector from one example", "variance across examples/hyperparams"],
    ["RePS (NeurIPS'25)", "compares vector/LoReFT/LoRA objectives", "needs tuning trick for stability"],
    ["PrOSV (ICML'26)", "trained steering factor + direction", "lr/init scale critical for stability"],
    ["IDEEA (EMNLP F'26)", "input-dependent CAA, 6 models", "different judges/split — not comparable"],
    ["Sun et al. (COLM'24)", "massive activations", "explains why raw CAA fails here"],
], widths=[3.2, 4.6, 4.6], fs=12.5, rowh=0.095)
done(fig, "literature", top=0.84)

# ---------------------------------------------------------------- 6. Table 1
fig = plt.figure(figsize=(13.33, 7.5))
fig.text(0.04, 0.95, "5 · Restructured Table 1 (current draft)", fontsize=21, fontweight="bold", color=INK, va="top")
fig.text(0.04, 0.035, "Status: done. MAST: acts on activations, weights unchanged, training needed. Two blocks: reported by RaLFiT vs re-run by us.",
         fontsize=12, color=INK2)
from PIL import Image  # noqa: E402
im = Image.open(TABLE1)
w_, h_ = im.size
im = im.crop((0, 0, w_, int(h_ * 0.715)))
ax = fig.add_axes([0.03, 0.1, 0.94, 0.78])
ax.imshow(im)
ax.axis("off")
fig.savefig(OUT / f"{len(SLIDES) + 1:02d}_table1.png", dpi=110)
SLIDES.append(fig)

# ---------------------------------------------------------------- 7. metric
fig, ax = new("6–7 · Two definitions of True × Info — we report both",
              "Status: done (§4 + Appendix B). Main tables use the product (RaLFiT / ITI convention); per-item is the original TruthfulQA definition.")
ax.axis("off")
ax.text(0.0, 0.97, "Product of rates (RaLFiT, TruthX, ITI's code):   T×I = mean(truthful) × mean(informative)", fontsize=14, transform=ax.transAxes, va="top")
ax.text(0.0, 0.89, "Per answer (Lin et al. 2022, '% true and informative'):   T∧I = mean(truthful AND informative)", fontsize=14, transform=ax.transAxes, va="top")
sub = fig.add_axes([0.08, 0.16, 0.85, 0.5])
meths = [("Unsteered", "baseline"), ("Raw CAA α=1", "caa_a1"), ("Direct vector (best lr)", "dvzero_lr2e-3"), ("MAST", "mast"), ("LoRA-DPO", "loradpo")]
xx = np.arange(len(meths))
sub.bar(xx - 0.18, [m(k)[0] for _, k in meths], 0.34, color="#4a3aa7", label="T×I (product)")
sub.bar(xx + 0.18, [m(k, "ti_conj")[0] for _, k in meths], 0.34, color="#e87ba4", label="T∧I (per item)")
for x, (_, k) in zip(xx, meths):
    sub.text(x - 0.18, m(k)[0] + 0.8, f"{m(k)[0]:.1f}", ha="center", fontsize=11)
    sub.text(x + 0.18, m(k, "ti_conj")[0] + 0.8, f"{m(k, 'ti_conj')[0]:.1f}", ha="center", fontsize=11)
sub.set_xticks(xx, [n for n, _ in meths])
sub.set_ylim(40, 90)
sub.legend(frameon=False, loc="upper left")
sub.set_ylabel("%")
fig.savefig(OUT / f"{len(SLIDES) + 1:02d}_metric.png", dpi=110)
SLIDES.append(fig)

# ---------------------------------------------------------------- 8. seeds
fig, ax = new("8 · Every method: 3 seeds × 2 folds",
              "Each dot is one seed×fold cell (≈408 test questions); bar = mean. Status: done for all re-run methods.")
meths = [("Unsteered", "baseline", C_BASE), ("Raw CAA α=1", "caa_a1", C_CAA), ("Raw CAA α=2", "caa_a2", C_CAA),
         ("Direct\n@5e-4", "dvzero_lr5e-4", C_DV), ("Direct\n@2e-3", "dvzero_lr2e-3", C_DV),
         ("MAST", "mast", C_MAST), ("LoRA-DPO", "loradpo", C_LORA)]
for x, (n, k, c) in enumerate(meths):
    vals = [v["ti_product"] for v in T[k]["cells"].values()]
    ax.bar(x, np.mean(vals), color=c, width=0.6, alpha=0.35)
    ax.scatter(x + np.linspace(-0.18, 0.18, len(vals)), vals, color=c, s=45, zorder=3, edgecolor="white")
ax.set_xticks(range(len(meths)), [n for n, _, _ in meths])
ax.set_ylabel("True × Info (%)")
ax.set_ylim(45, 88)
done(fig, "seeds")

# ---------------------------------------------------------------- 9. Rev 3 generality
fig, axs = new("9 · Reviewer 3: generalisation experiments (already run)",
               "Status: done; 4 extra category-hold-out seeds being scored. Left/middle: True∧Info / True×Info; right: exact match.", ncols=3)
ch = []
for fold in ("A", "B"):
    ch.append([100 * judged(O / f"cathold_fold_{fold}/{v}/{s}/gpt_judge_results.json")[1]["truth_and_info_accuracy"]
               for v, s in (("baseline", "scale_0.00"), ("mlp_mc", "scale_1.00"))])
for k, (lab, c) in enumerate((("unsteered", C_BASE), ("MAST", C_MAST))):
    axs[0].bar(np.arange(2) + (k - 0.5) * 0.36, [v[k] for v in ch], 0.34, color=c, label=lab)
axs[0].set_xticks([0, 1], ["fold A", "fold B"])
axs[0].set_title("Unseen categories (seed 42)", fontsize=13)
axs[0].set_ylim(0, 100)
axs[0].legend(frameon=False, fontsize=11)
g = {}
for lab, v, s in (("unsteered", "baseline", "scale_0.00"), ("raw CAA", "steered", "scale_1.00"), ("MAST", "mlp_mc", "scale_1.00")):
    rr = judged(O / f"g4b_bn8_full/{v}/{s}/gpt_judge_results.json")[0] + judged(O / f"rcv_g4bmain_s42/fold2/{v}/{s}/gpt_judge_results.json")[0]
    g[lab] = 100 * np.mean([r["truth_judgment"] == "yes" for r in rr]) * np.mean([r["info_judgment"] == "yes" for r in rr])
axs[1].bar(range(3), list(g.values()), color=[C_BASE, C_CAA, C_MAST], width=0.6)
axs[1].set_xticks(range(3), list(g))
axs[1].set_ylim(0, 100)
axs[1].set_title("Gemma-3-4B-IT (2-fold, seed 42)", fontsize=13)
pq = json.loads((O / "multiseed/seed_42/transfer_popqa.json").read_text())["variants"]
nq = json.loads((O / "multiseed/seed_42/transfer_nq.json").read_text())["variants"]
axs[2].bar(np.arange(2) - 0.18, [pq["baseline"]["em_contains"], nq["baseline"]["em_contains"]], 0.34, color=C_BASE, label="unsteered")
axs[2].bar(np.arange(2) + 0.18, [pq["steered"]["em_contains"], nq["steered"]["em_contains"]], 0.34, color=C_MAST, label="+ vector")
axs[2].set_xticks([0, 1], ["PopQA", "NQ-open"])
axs[2].set_title("Knowledge QA (seed 42)", fontsize=13)
axs[2].legend(frameon=False, fontsize=11, loc="upper right")
axs[2].set_ylim(0, 42)
for ax in axs:
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width() / 2, p.get_height() + 1, f"{p.get_height():.0f}", ha="center", fontsize=11)
done(fig, "rev3")

# ---------------------------------------------------------------- 10. protocol
fig, axs = new("10 · Protocol aligned with RaLFiT", "Status: done (§4). Right: our re-runs vs RaLFiT's reported numbers (same judge family, 2-fold CV).",
               ncols=2, gridspec_kw={"width_ratios": [1.5, 1]})
table(axs[0], ["", "RaLFiT", "Ours"], [
    ["Split", "2-fold CV, all 817", "2-fold CV, all 817"],
    ["Repetitions", "1 run", "3 seeds"],
    ["Judges", "fine-tuned GPT-4o-mini", "fine-tuned GPT-4o-mini"],
    ["Metric", "Truth × Info", "both aggregations"],
    ["Generation prompt", "not stated", "6-shot TruthfulQA QA (as ITI)"],
    ["MC1 / MC2", "lm-eval", "lm-eval, test half"],
    ["LoRA baseline", "r8 on W_O, W_down, DPO", "same (5.96M params)"],
], widths=[2.2, 2.6, 2.8], fs=12.5, rowh=0.11)
labs = ["unsteered", "LoRA-DPO", "RaLFiT"]
axs[1].bar(np.arange(3) - 0.18, [54.56, 76.54, 77.40], 0.34, color=C_CAA, label="reported by RaLFiT")
axs[1].bar(np.arange(2) + 0.18, [m("baseline")[0], m("loradpo")[0]], 0.34, color=[C_BASE, C_LORA], label="our re-run")
axs[1].set_xticks(range(3), labs)
axs[1].set_ylim(40, 90)
axs[1].legend(frameon=False, fontsize=11, loc="upper left")
axs[1].text(2.0, 42, "RaLFiT: no public code", ha="center", fontsize=10, color=INK2)
for p in axs[1].patches:
    axs[1].text(p.get_x() + p.get_width() / 2, p.get_height() + 0.8, f"{p.get_height():.1f}", ha="center", fontsize=10.5)
done(fig, "protocol", wspace=0.15)

# ---------------------------------------------------------------- 11. open questions
fig, ax = new("Open questions for today", None)
ax.axis("off")
for i, ln in enumerate([
    "1.  MAST ≈ direct vector on LLaMA (each at its best lr). Keep MAST as the method and say so,",
    "     or reframe around supervised steering vectors?",
    "2.  Gemma: the bare vector fails at LLaMA's lrs; MAST transfers unchanged.",
    "     Worth ~1 GPU-day to test 2–3 more models?",
    "3.  MAST is 2.9 pts below LoRA-DPO (same data). Re-implement RaLFiT itself",
    "     (~½ day; no public code)?",
    "4.  Interpretability findings (appendix A1–A2): include, or keep for the thesis?",
]):
    ax.text(0.0, 0.92 - i * 0.1, ln, fontsize=15, transform=ax.transAxes, va="top", color=INK)
done(fig, "questions")

# ---------------------------------------------------------------- A1. geometry + CAA
def vec(p):
    return torch.load(p, map_location="cpu").float().flatten()


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


pairs = {"MAST vs direct (zero init)": [], "MAST vs direct (CAA init)": [], "direct (zero) vs raw CAA": [], "MAST vs raw CAA": []}
share = {"raw CAA": [], "MAST": [], "direct (zero init)": []}
for s, f in CELLS:
    mv, z = vec(O / f"rcv_main_s{s}/fold{f}/vectors/v_mlp_mc.pt"), vec(O / f"rcv_dvzero_lr2e-3_s{s}/fold{f}/vectors/optimized_vector.pt")
    k, c = vec(O / f"rcv_dvcaa_lr2e-3_s{s}/fold{f}/vectors/optimized_vector.pt"), vec(O / f"rcv_main_s{s}/fold{f}/vectors/v_steered.pt")
    for name, (a, b) in zip(pairs, ((mv, z), (mv, k), (z, c), (mv, c))):
        pairs[name].append(cos(a, b))
    for name, v in zip(share, (c, mv, z)):
        share[name].append(100 * float((v[[1415, 2533]] ** 2).sum() / (v ** 2).sum()))
fig, axs = new("Appendix A1 · Exploratory: vector geometry", "6 seed×fold cells. Left: cosine between vectors trained on the same data. "
               "Right: share of the vector's squared norm in dims 1415 and 2533 (LLaMA-2's massive-activation dims, Sun et al. 2024).", ncols=2)
names = list(pairs)
axs[0].barh(range(4)[::-1], [np.mean(pairs[n]) for n in names], xerr=[np.std(pairs[n]) for n in names],
            color=[C_MAST, C_MAST, C_CAA, C_CAA], height=0.55, capsize=4)
axs[0].set_yticks(range(4)[::-1], names)
axs[0].set_xlim(0, 1)
axs[0].set_xlabel("cosine similarity")
axs[1].bar(range(3), [np.mean(v) for v in share.values()], color=[C_CAA, C_MAST, C_DV], width=0.6)
for x, v in enumerate(share.values()):
    axs[1].scatter(np.full(6, x) + np.linspace(-0.15, 0.15, 6), v, color=INK, s=20, zorder=3)
axs[1].set_xticks(range(3), list(share))
axs[1].set_ylabel("% of squared norm")
axs[1].set_ylim(0, 105)
done(fig, "A1_geometry", left=0.22, wspace=0.35)

# ---------------------------------------------------------------- A2. hedging
H = re.compile(r"\b(no comment|i don'?t know|i do not know|i cannot|i can'?t|not sure|i'?m not able|unable to|"
               r"there is no (?:scientific )?evidence|it is not possible to|no one knows|depends)\b", re.I)
hm = [("unsteered", "main", "baseline/scale_0.00", C_BASE), ("direct (best lr)", "dvzero_lr2e-3", "mlp_mc/scale_1.00", C_DV),
      ("MAST", "main", "mlp_mc/scale_1.00", C_MAST), ("LoRA-DPO", "loradpo", "mlp_mc/scale_1.00", C_LORA)]
hed, sub_ = [], []
for _, meth, sp, _ in hm:
    h, t = [], []
    for s, f in CELLS:
        for r in judged(O / f"rcv_{meth}_s{s}/fold{f}/{sp}/gpt_judge_results.json")[0]:
            hh = bool(H.search(r["generated_clean"]))
            h.append(hh)
            if not hh:
                t.append(r["truth_judgment"] == "yes" and r["info_judgment"] == "yes")
    hed.append(100 * np.mean(h))
    sub_.append(100 * np.mean(t))
fig, axs = new("Appendix A2 · Exploratory: hedging", "Hedged = matches a refusal/uncertainty phrase list (indicative). 6 cells pooled.", ncols=2)
for ax, vals, t in ((axs[0], hed, "hedged answers (%)"), (axs[1], sub_, "Truth∧Info among non-hedged answers (%)")):
    ax.bar(range(4), vals, color=[h[3] for h in hm], width=0.6)
    ax.set_xticks(range(4), [h[0] for h in hm])
    ax.set_title(t, fontsize=13)
    for x, v in enumerate(vals):
        ax.text(x, v + 1, f"{v:.0f}", ha="center", fontsize=11)
done(fig, "A2_hedging")

with PdfPages(OUT / "deck.pdf") as pdf:
    for f in SLIDES:
        pdf.savefig(f)
print(f"{len(SLIDES)} slides -> {OUT / 'deck.pdf'}")
