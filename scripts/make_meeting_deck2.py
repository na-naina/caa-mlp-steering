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
        fig.text(0.04, 0.025, textwrap.fill(status, 140), fontsize=11.5, color=INK2, va="bottom")
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
table(ax, ["#", "Item", "Status"], [
    ["1", "Per-category breakdown (Rev. 74Zo)", "done"],
    ["2", "Setup aligned with the literature (fair comparison)", "done"],
    ["3", "Direct-vector baseline across lrs; prior work; always vs sometimes", "done (LLaMA, Gemma); + loss vs BiPO"],
    ["4", "Recent literature", "done, verified"],
    ["5", "Table 1 restructured + 'training' column", "done"],
    ["6", "T×I definition; alignment with original paper and RaLFiT", "done — main metric to decide"],
    ["7", "Alternative metric in appendix", "done"],
    ["8", "Multiple seeds for every method", "done"],
    ["9", "Reviewer 3: add the already-run experiments", "done; Qwen3-4B running"],
    ["10", "Aligned with RaLFiT's protocol", "done"],
], widths=[0.4, 6.4, 3.6], fs=13, rowh=0.083)
done(fig, "status", top=0.84)

# ---------------------------------------------------------------- 1. categories
cats = R["categories"]
cm = {c: v for c, v in cats["mast"].items() if v["n"] >= 30}
rows = sorted(((c, cats["baseline"][c]["ti_conj"], cm[c]["ti_conj"], cm[c]["n"] // 3) for c in cm), key=lambda t: t[2] - t[1])
fig, ax = new("Per-category results, Reviewer 74Zo (item 1)",
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

# ---------------------------------------------------------------- 2·8·10 setup
fig, axs = new("Same setup for every method we run (items 2, 8, 10)",
               "Instead of copying other papers' numbers, we re-run each baseline ourselves under one protocol (RaLFiT's), so every comparison is apples to apples. "
               "Right: each dot is one seed×fold run; bar = mean.", ncols=2, gridspec_kw={"width_ratios": [1.25, 1]})
table(axs[0], ["", "Every method (ours & re-run baselines)"], [
    ["Model / layer", "LLaMA-2-7B-Chat, layer 8 (as ITI, TruthX, BiPO, RaLFiT)"],
    ["Test protocol", "2-fold CV: each of 817 Qs answered once by a model"],
    ["", "that never trained on it (= RaLFiT, ITI, TruthX)"],
    ["Seeds", "the random 2-fold split is redrawn 3 times"],
    ["", "→ 6 runs per method (RaLFiT: 1 run)"],
    ["Training data", "identical questions & answer pairs per fold"],
    ["Prompt / decoding", "TruthfulQA 6-shot QA prompt, T = 0.3"],
    ["Judges", "same fine-tuned GPT-4o-mini judges as RaLFiT"],
    ["Calibration", "unsteered 56.5 (RaLFiT: 54.6); LoRA-DPO 80.1 (76.5)"],
], widths=[2.2, 5.6], fs=12.5, rowh=0.095)
meths = [("Unsteered", "baseline", C_BASE), ("Raw\nCAA", "caa_a1", C_CAA), ("Direct\n@5e-4", "dvzero_lr5e-4", C_DV),
         ("Direct\n@2e-3", "dvzero_lr2e-3", C_DV), ("MAST", "mast", C_MAST), ("LoRA-\nDPO", "loradpo", C_LORA)]
for x, (n, k, c) in enumerate(meths):
    vals = [v["ti_product"] for v in T[k]["cells"].values()]
    axs[1].bar(x, np.mean(vals), color=c, width=0.6, alpha=0.35)
    axs[1].scatter(x + np.linspace(-0.18, 0.18, len(vals)), vals, color=c, s=40, zorder=3, edgecolor="white")
axs[1].set_xticks(range(len(meths)), [n for n, _, _ in meths], fontsize=11)
axs[1].set_ylabel("True × Info (%)")
axs[1].set_ylim(45, 88)
done(fig, "setup", wspace=0.12)

# ---------------------------------------------------------------- 3. direct vector (LLaMA)
fig, ax = new("Direct vector vs MAST across learning rates, LLaMA-2-7B-Chat (item 3)",
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

# ---------------------------------------------------------------- 3b. Gemma lr sweep
def oti(p):
    st = judged(p / "gpt_judge_results.json")[1]
    return 100 * st["truth_accuracy"] * st["info_accuracy"]
gl = []
for lr in ("5e-4", "2e-3", "5e-3", "3e-2", "1e-1", "3e-1"):
    d = O / f"rg4b_dvzero_lr{lr}_s42/fold1"
    if (d / "mlp_mc/scale_1.00/gpt_judge_results.json").exists():
        gl.append((float(lr), oti(d / "mlp_mc/scale_1.00"), json.loads((d / "meta.json").read_text())["v_final_norm"]))
gmast = oti(O / "g4b_bn8_full/mlp_mc/scale_1.00")
fig, ax = new("Direct vector vs MAST on Gemma-3-4B-IT (item 3, preliminary)",
              "Seed 42, fold 1, GPT-4o-mini judges. Labels: norm of the learned vector. MAST uses its LLaMA settings unchanged (lr 5e-4); "
              "Gemma's residual activations are ~50× larger than LLaMA's (||v_CAA|| ≈ 134 vs 2.5).")
ax.plot([g[0] for g in gl], [g[1] for g in gl], color=C_DV, marker="s", ms=10, lw=2.2, label="direct vector (zero init)")
for x, y, n in gl:
    ax.text(x, y + 1.6, f"‖v‖ = {n:.0f}", ha="center", fontsize=10.5, color=INK2)
ax.axhline(gmast, color=C_MAST, lw=2.2, label=f"MAST, lr 5e-4 ({gmast:.1f})")
ax.axhline(oti(O / "g4b_bn8_full/baseline/scale_0.00"), color=C_BASE, lw=1.5, ls=":", label="unsteered")
ax.set_xscale("log")
ax.set_xlabel("learning rate (direct vector)")
ax.set_ylabel("True × Info (%)")
ax.set_ylim(45, 100)
ax.legend(frameon=False, fontsize=12, loc="upper left")
done(fig, "direct_vector_gemma")

# ---------------------------------------------------------------- 3c. same settings across models
best = {"LLaMA-2-7B-Chat": (O / "rcv_dvzero_lr2e-3_s42/fold1/mlp_mc/scale_1.00", "2e-3"),
        "Gemma-3-4B-IT": (O / "rg4b_dvzero_lr3e-1_s42/fold1/mlp_mc/scale_1.00", "3e-1")}
rows = [("MAST, lr 5e-4 (both models)", O / "rcv_main_s42/fold1/mlp_mc/scale_1.00", O / "g4b_bn8_full/mlp_mc/scale_1.00", C_MAST),
        ("direct vector, lr 2e-3 (both models)", O / "rcv_dvzero_lr2e-3_s42/fold1/mlp_mc/scale_1.00", O / "rg4b_dvzero_lr2e-3_s42/fold1/mlp_mc/scale_1.00", "#f2a07c"),
        ("direct vector, best lr per model (2e-3 / 3e-1)", best["LLaMA-2-7B-Chat"][0], best["Gemma-3-4B-IT"][0], C_DV),
        ("direct vector × ||v_CAA||, lr 8e-4 (both models)", O / "rcv_dvscaled_lr8e-4_s42/fold1/mlp_mc/scale_1.00", O / "rg4b_dvscaled_lr8e-4_s42/fold1/mlp_mc/scale_1.00", "#4a3aa7")]
fig, ax = new("Same hyperparameters on two models (item 3, preliminary)",
              "Seed 42, fold 1, GPT-4o-mini judges. 'Direct vector × ||v_CAA||': v = s·u with s = norm of that model's CAA vector (fixed), u trained; "
              "one lr then gives a step size proportional to the model's activation scale.")
xx = np.arange(2)
wd = 0.2
for k, (lab, pl, pg, c) in enumerate(rows):
    vals = [oti(pl), oti(pg)]
    ax.bar(xx + (k - 1.5) * wd, vals, wd * 0.92, color=c, label=lab)
    for x, v in zip(xx, vals):
        ax.text(x + (k - 1.5) * wd, v + 1, f"{v:.1f}", ha="center", fontsize=11)
ax.axhline(0, color=INK2, lw=0.5)
ax.set_xticks(xx, list(best))
ax.set_ylim(40, 105)
ax.set_ylabel("True × Info (%)")
ax.legend(frameon=False, fontsize=11.5, loc="upper left")
done(fig, "same_settings")

# ---------------------------------------------------------------- 3d. loss
fig, ax = new("Our margin loss vs BiPO's preference loss (item 3)",
              "Direct vector (zero init), identical data, steps and generation; only the loss differs. LLaMA-2-7B-Chat, seed 42, both folds, GPT-4o-mini judges. "
              "Remaining BiPO seeds are queued.")
bp = []
for lr in ("5e-4", "2e-3"):
    for lab, pref, c in (("BiPO loss", "rcv_bipo_lr", "#e87ba4"), ("our margin loss", "rcv_dvzero_lr", C_DV)):
        vals = [oti(O / f"{pref}{lr}_s42/fold{f}/mlp_mc/scale_1.00") for f in (1, 2)
                if (O / f"{pref}{lr}_s42/fold{f}/mlp_mc/scale_1.00/gpt_judge_results.json").exists()]
        bp.append((lr, lab, c, vals))
for k, (lr, lab, c, vals) in enumerate(bp):
    x = k + (k // 2) * 0.6
    ax.bar(x, np.mean(vals), color=c, width=0.8, label=lab if k < 2 else None)
    ax.scatter(np.full(len(vals), x) + np.linspace(-0.12, 0.12, len(vals)), vals, color=INK, s=28, zorder=3)
    ax.text(x, np.mean(vals) + 1.2, f"{np.mean(vals):.1f}", ha="center", fontsize=12)
ax.set_xticks([0.5, 3.1], ["lr 5e-4", "lr 2e-3"])
ax.set_ylim(50, 92)
ax.set_ylabel("True × Info (%)  (bar = mean of 2 folds, dots = folds)")
ax.legend(frameon=False, fontsize=12, loc="upper left")
done(fig, "loss")

# ---------------------------------------------------------------- 4. literature
fig, ax = new("Literature that drives our decisions (item 4)",
              "Status: done; full survey incl. 2026 papers (IDEEA, RePS, PrOSV, HyperSteer) in docs/lit_update_oct2026.md, claims verified against the papers.")
table(ax, ["Paper", "Decision it informs"], [
    ["Lin et al. 2022 (TruthfulQA)", "original metric definition (items 6–7)"],
    ["RaLFiT, Li et al. 2025 (ACL Findings)", "strongest prior result → our protocol, judges, LoRA-DPO baseline"],
    ["BiPO, Cao et al. 2024 (NeurIPS)", "closest prior direct-vector method; MC only → our open-ended comparison is new"],
    ["LoFiT / RED (2024)", "learned offsets already work → training vectors is not new per se"],
    ["Dunefsky & Cohan 2025; RePS 2025", "optimised vectors are hyperparameter-sensitive → supports our lr analysis"],
    ["Sun et al. 2024 (massive activations)", "explains why raw CAA fails on this model"],
], widths=[4.2, 8.6], fs=13.5, rowh=0.11)
done(fig, "literature", top=0.84)

# ---------------------------------------------------------------- 6. Table 1
fig = plt.figure(figsize=(13.33, 7.5))
fig.text(0.04, 0.95, "Restructured Table 1, current draft (item 5)", fontsize=21, fontweight="bold", color=INK, va="top")
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
fig, ax = new("Which True × Info definition? (items 6–7)",
              "Proposal: main tables use the product (as RaLFiT and the SOTA tables, so rows are comparable); the original paper's per-answer definition goes to the appendix. Decision for today.")
ax.axis("off")
ax.text(0.0, 0.97, "Product of rates (RaLFiT, TruthX, ITI's code — the SOTA tables):   mean(truthful) × mean(informative)", fontsize=14, transform=ax.transAxes, va="top")
ax.text(0.0, 0.89, "Per answer (Lin et al. 2022, the original TruthfulQA paper, '% true and informative'):   mean(truthful AND informative)", fontsize=14, transform=ax.transAxes, va="top")
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

# ---------------------------------------------------------------- 9a method beyond TruthfulQA/LLaMA
fig, axs = new("Does the method work beyond TruthfulQA and LLaMA? (item 9, Reviewer 3)",
               "Each panel trains the recipe on the target model/task itself (seed 42). Positive on categories, model and a second behaviour; "
               "negative on knowledge QA (PopQA: unsteered greedy vs trained vector sampled at T=0.3 — indicative). Not yet run: HaluEval (candidate for Friday).", ncols=4)
ch = []
for fold in ("A", "B"):
    ch.append([100 * judged(O / f"cathold_fold_{fold}/{v}/{s_}/gpt_judge_results.json")[1]["truth_and_info_accuracy"]
               for v, s_ in (("baseline", "scale_0.00"), ("mlp_mc", "scale_1.00"))])
for k, (lab, c) in enumerate((("unsteered", C_BASE), ("MAST", C_MAST))):
    axs[0].bar(np.arange(2) + (k - 0.5) * 0.36, [v[k] for v in ch], 0.34, color=c, label=lab)
axs[0].set_xticks([0, 1], ["fold A", "fold B"])
axs[0].set_title("Unseen TruthfulQA categories\n(Truth∧Info)", fontsize=12)
axs[0].set_ylim(0, 100)
axs[0].legend(frameon=False, fontsize=10)
g = {}
for lab, v, s_ in (("unsteered", "baseline", "scale_0.00"), ("raw CAA", "steered", "scale_1.00"), ("MAST", "mlp_mc", "scale_1.00")):
    rr = judged(O / f"g4b_bn8_full/{v}/{s_}/gpt_judge_results.json")[0] + judged(O / f"rcv_g4bmain_s42/fold2/{v}/{s_}/gpt_judge_results.json")[0]
    g[lab] = 100 * np.mean([r["truth_judgment"] == "yes" for r in rr]) * np.mean([r["info_judgment"] == "yes" for r in rr])
axs[1].bar(range(3), list(g.values()), color=[C_BASE, C_CAA, C_MAST], width=0.6)
axs[1].set_xticks(range(3), list(g))
axs[1].set_ylim(0, 100)
axs[1].set_title("Another model: Gemma-3-4B-IT\n(True×Info, 2-fold)", fontsize=12)
abd = json.loads((O / "box5090_final/halluc_vector/ab_eval_hallucination.json").read_text())["scales"]
abv = [abd[k]["behavior_match_rate"] for k in ("-1.0", "0.0", "1.0")]
axs[2].bar(range(3), abv, color=[C_MAST, C_BASE, C_DV], width=0.6)
axs[2].set_xticks(range(3), ["α = −1", "unsteered", "α = +1"])
axs[2].set_ylim(0, 110)
axs[2].set_title("Another behaviour: hallucination\n(% choosing hallucinated answer, n=50)", fontsize=12)
pq = json.loads((O / "multiseed/seed_42/transfer_popqa.json").read_text())["variants"]
axs[3].bar([0, 1], [pq["baseline"]["em_contains"], 26.3], color=[C_BASE, C_MAST], width=0.6)
axs[3].set_xticks([0, 1], ["unsteered", "vector trained\non PopQA"])
axs[3].set_ylim(0, 42)
axs[3].set_title("Knowledge task: PopQA\n(exact match, n=1000)", fontsize=12)
for ax in axs:
    for p_ in ax.patches:
        ax.text(p_.get_x() + p_.get_width() / 2, p_.get_height() + 1, f"{p_.get_height():.0f}", ha="center", fontsize=11)
done(fig, "method_beyond")

# ---------------------------------------------------------------- 9a2 Qwen3-4B
qd = O / "rq4b_main_L14/fold1"
qrows = [("unsteered", qd / "baseline/scale_0.00", C_BASE), ("raw CAA", qd / "steered/scale_1.00", C_CAA),
         ("MAST", qd / "mlp_mc/scale_1.00", C_MAST)]
fig, ax = new("A third model family: Qwen3-4B (item 9, preliminary)",
              "Same recipe and hyperparameters as LLaMA (k=8, lr 5e-4, 100 steps); layer 14 chosen from training loss only (vs layer 9). "
              "Seed 42, fold 1, GPT-4o-mini judges. Plain few-shot prompt, no chat template.")
for x, (lab, p_, c) in enumerate(qrows):
    f = p_ / "gpt_judge_results.json"
    if f.exists():
        st = judged(f)[1]
        t, i_ = 100 * st["truth_accuracy"], 100 * st["info_accuracy"]
        ax.bar(x, t * i_ / 100, color=c, width=0.6)
        ax.text(x, t * i_ / 100 + 1, f"{t * i_ / 100:.1f}\n(Truth {t:.0f}, Info {i_:.0f})", ha="center", fontsize=12)
    else:
        ax.text(x, 45, "running", ha="center", fontsize=13, color=INK2)
ax.set_xticks(range(3), [r[0] for r in qrows])
ax.set_ylim(0, 105)
ax.set_ylabel("True × Info (%)")
done(fig, "qwen")

# ---------------------------------------------------------------- 9b capability preservation
fig, axs = new("Capability preservation: TruthfulQA vector on other tasks (item 9)",
               "Same LLaMA-2-7B-Chat vector (seed 42), no retraining. Left: zero-shot lm-eval accuracy. Right: open QA exact match "
               "(1,000 questions each). RaLFiT reports roughly neutral changes on ARC/HellaSwag/MMLU (+4.4/+1.2/−0.5, different protocol).", ncols=2)
cap = json.loads((O / "coherence_bn8_seed42/coherence_results.json").read_text())
def capv(x):
    mm = [x[k]["acc"] for k in x if k.startswith("mmlu_") and x[k]["acc"] is not None]
    return [100 * x["arc_easy"]["acc"], 100 * x["arc_challenge"]["acc_norm"], 100 * x["hellaswag"]["acc_norm"], 100 * np.mean(mm)]
b_, s2 = capv(cap["baseline"]), capv(cap["steered"])
xx = np.arange(4)
axs[0].bar(xx - 0.18, b_, 0.34, color=C_BASE, label="unsteered")
axs[0].bar(xx + 0.18, s2, 0.34, color=C_MAST, label="+ TruthfulQA vector")
axs[0].set_xticks(xx, ["ARC-Easy", "ARC-Chall.", "HellaSwag", "MMLU\n(subject mean)"])
axs[0].set_ylim(0, 95)
axs[0].legend(frameon=False, fontsize=11)
nq = json.loads((O / "multiseed/seed_42/transfer_nq.json").read_text())["variants"]
axs[1].bar(np.arange(2) - 0.18, [pq["baseline"]["em_contains"], nq["baseline"]["em_contains"]], 0.34, color=C_BASE, label="unsteered")
axs[1].bar(np.arange(2) + 0.18, [pq["steered"]["em_contains"], nq["steered"]["em_contains"]], 0.34, color=C_MAST, label="+ TruthfulQA vector")
axs[1].set_xticks([0, 1], ["PopQA", "NQ-open"])
axs[1].set_ylim(0, 42)
axs[1].legend(frameon=False, fontsize=11)
for ax in axs:
    for p_ in ax.patches:
        ax.text(p_.get_x() + p_.get_width() / 2, p_.get_height() + 0.8, f"{p_.get_height():.1f}", ha="center", fontsize=10.5)
done(fig, "capability")

# ---------------------------------------------------------------- 11. talking points
fig, ax = new("Where this leaves us", None)
ax.axis("off")
lines = [
    ("h", "1.  All review points are addressed (status slide)."),
    ("t", "     But the answers make the paper weaker: the MLP behaves mostly as a step-size scaler — a directly optimised vector"),
    ("t", "     with a learning rate matched to the activation scale ties MAST on LLaMA and is ~2 pts behind on Gemma (1 seed)."),
    ("t", "     The CAA starting point matters less than reported: noise input is ~6 pts below MAST over 3 seeds (was ~17 with 1 seed)."),
    ("h", "2.  Realistic ways to reframe"),
    ("t", "     a) The loss: our margin loss vs BiPO's — small edge so far (seed 42 only); a finding only if it holds across seeds and models."),
    ("t", "     b) An empirical study: one supervised vector ≈ 88% of LoRA-DPO's gain without weight changes; why CAA fails; limits."),
    ("t", "     c) Interpretability (what the supervised direction is, why CAA misses it) — promising but needs much more work."),
    ("h", "3.  One more meeting this Friday: what we present, and whether we submit on 12 Oct."),
    ("t", "     By then: loss comparison on all seeds, Qwen3-4B, and whichever of (a)–(c) we pick today to push."),
    ("h", "4.  API key for the fine-tuned GPT judges"),
    ("t", "     Spent this round ≈ $2 (≈ $0.02 per 408-answer run). Two options for the rest of the cycle:"),
    ("t", "     conservative — GPT judges only for core runs: ≈ $3.5–5 in total;"),
    ("t", "     everything on GPT judges (one judge for every number, cleaner bookkeeping): ≈ $6–9 in total."),
]
y = 0.97
for kind, ln in lines:
    ax.text(0.0, y, ln.strip() if kind == "h" else ln, fontsize=16 if kind == "h" else 13,
            fontweight="bold" if kind == "h" else "normal", color=INK if kind == "h" else INK2, transform=ax.transAxes, va="top")
    y -= 0.095 if kind == "h" else 0.075
done(fig, "talking_points")

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

# ---------------------------------------------------------------- A3. interventions
X = lambda n: O / f"{n}/fold1/mlp_mc/scale_1.00"  # noqa: E731
groups = [
    ("reference", [("Unsteered", O / "rcv_main_s42/fold1/baseline/scale_0.00", C_BASE),
                   ("MAST", O / "rcv_main_s42/fold1/mlp_mc/scale_1.00", C_MAST),
                   ("Direct vector (best lr)", O / "rcv_dvzero_lr2e-3_s42/fold1/mlp_mc/scale_1.00", C_DV),
                   ("LoRA-DPO", O / "rcv_loradpo_s42/fold1/mlp_mc/scale_1.00", C_LORA)]),
    ("remove / subtract direction", [
                   ("Direct vector subtracted (α = −1)", X("rx_dvz_neg"), C_DV),
                   ("Unsteered, direction removed at all layers", X("rx_base_ablall_dvz"), C_BASE),
                   ("LoRA-DPO, direction removed at layer 8", X("rx_lora_abl8_dvz"), C_LORA),
                   ("LoRA-DPO, direction removed at all layers", X("rx_lora_ablall_dvz"), C_LORA)]),
    ("other vectors", [
                   ("Average of 3 supervised vectors", X("rx_consensus"), C_MAST),
                   ("LoRA's activation shift as a vector (α = 2)", X("rx_distilled_a2"), C_LORA),
                   ("CAA, answer-token pooling (no artefact)", X("rx_caa_answer"), C_CAA),
                   ("Geometry-of-Truth direction (α = 1)", X("rx_gotdir_a1"), C_CAA)]),
]
fig, ax = new("Appendix A3 · Exploratory: interventions (seed 42, fold 1)",
              "GPT-4o-mini judges for every row (408 test questions). 'Direction' = the zero-init supervised vector, which has no massive-activation component.")
y, ticks, labs = 0, [], []
for gname, rows in groups[::-1]:
    for lab, p, c in rows[::-1]:
        f = p / "gpt_judge_results.json"
        if not f.exists():
            continue
        st = judged(f)[1]
        t, i_ = 100 * st["truth_accuracy"], 100 * st.get("info_accuracy", float("nan"))
        ax.barh(y, t * i_ / 100, color=c, height=0.65)
        ax.text(t * i_ / 100 + 1, y, f"{t * i_ / 100:.1f}   (Truth {t:.0f}, Info {i_:.0f})", va="center", fontsize=10.5)
        ticks.append(y); labs.append(lab); y += 1
    ax.text(-0.01, y - 0.35, gname, transform=ax.get_yaxis_transform(), ha="right", fontsize=10.5, fontweight="bold", color=INK2)
    y += 0.8
ax.set_yticks(ticks, labs, fontsize=10.5)
ax.set_xlim(0, 105)
ax.set_xlabel("True × Info (%)")
done(fig, "A3_interventions", left=0.3)

with PdfPages(OUT / "deck.pdf") as pdf:
    for f in SLIDES:
        pdf.savefig(f)
print(f"{len(SLIDES)} slides -> {OUT / 'deck.pdf'}")
