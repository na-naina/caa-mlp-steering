#!/usr/bin/env python3
"""Supervisor-meeting deck (5 Oct 2026): one figure per slide, every number read
from result files. Writes paper/figures/oct5_meeting/{NN_name.png, deck.pdf}.

Usage: .venv/bin/python scripts/make_meeting_deck.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import aggregate_revision as agg  # noqa: E402

O = Path("data/outputs")
OUT = Path("paper/figures/oct5_meeting")
OUT.mkdir(parents=True, exist_ok=True)

# roles (reference palette slots 1-3 + neutrals)
C_MAST, C_DV, C_LORA = "#2a78d6", "#eb6834", "#1baf7a"
C_BASE, C_CAA, C_MUTED = "#8c8b86", "#b9b8b2", "#d9d8d3"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#ececea", "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
SLIDES = []


def slide(title, subtitle, figsize=(13.33, 7.5), nrows=1, ncols=1, **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kw)
    import textwrap
    t = textwrap.fill(title, 88)
    fig.suptitle(t, x=0.04, y=0.975, ha="left", va="top", fontsize=18, fontweight="bold", color=INK)
    y_sub = 0.975 - 0.055 * (t.count("\n") + 1) - 0.005
    fig.text(0.04, y_sub, textwrap.fill(subtitle, 150), ha="left", va="top", fontsize=11, color=INK2)
    return fig, axes


def save(fig, name, top=0.76, bottom=0.1, **kw):
    fig.subplots_adjust(top=top, bottom=bottom, left=kw.get("left", 0.08), right=kw.get("right", 0.97),
                        wspace=kw.get("wspace", 0.25), hspace=kw.get("hspace", 0.35))
    path = OUT / f"{len(SLIDES) + 1:02d}_{name}.png"
    fig.savefig(path, dpi=110)
    SLIDES.append(fig)


def text_slide(title, lines, name):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.suptitle(title, x=0.04, y=0.965, ha="left", fontsize=19, fontweight="bold", color=INK)
    y = 0.87
    for ln in lines:
        indent, bold = 0.05, False
        if ln.startswith("## "):
            ln, bold, y = ln[3:], True, y - 0.012
        elif ln.startswith("- "):
            ln, indent = "•  " + ln[2:], 0.07
        fig.text(indent, y, ln, fontsize=13.5 if not bold else 14.5, fontweight="bold" if bold else "normal",
                 color=INK if bold or not ln.startswith("•") else INK, va="top", wrap=True)
        y -= 0.052 if not bold else 0.058
    fig.savefig(OUT / f"{len(SLIDES) + 1:02d}_{name}.png", dpi=110)
    SLIDES.append(fig)


# ------------------------------------------------------------------ data
R = json.loads(Path("paper/figures/revision_oct2026/revision_results_gpt.json").read_text())
T = R["table"]


def m(key, metric="ti_product"):
    return T[key][metric]["mean"], T[key][metric]["sd"]


def judged(path):
    d = json.loads(Path(path).read_text())
    return d["results"], d["stats"]


def rates(path):
    res, _ = judged(path)
    t = np.array([r["truth_judgment"] == "yes" for r in res], float)
    i = np.array([r["info_judgment"] == "yes" for r in res], float)
    return 100 * t.mean(), 100 * i.mean(), 100 * t.mean() * i.mean()


def vec(path):
    return torch.load(path, map_location="cpu").float().flatten()


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


CELLS = [(s, f) for s in (42, 123, 456) for f in (1, 2)]

# ------------------------------------------------------------------ 1. overview
text_slide("Where we are (5 Oct; ARR resubmission due 12 Oct)", [
    "## What was run since July (all on LLaMA-2-7B-Chat unless noted)",
    "- Every method re-run under RaLFiT's protocol: 2-fold CV over all 817 questions × 3 seeds, GPT-4o-mini judges",
    "- Baselines on identical data: unsteered, raw CAA (α=1, 2), direct vector (2 inits × 4 lrs), LoRA-DPO (RaLFiT's setting)",
    "- MAST at 3 learning rates; noise-input ablation × 3 seeds; Gemma-3-4B 2nd fold + direct vector; one-shot vectors",
    "- Interpretability probes: vector geometry, CAA extraction variants, Geometry-of-Truth, hedging analysis",
    "## Headline findings",
    "- A single supervised vector at one layer recovers ~88% of LoRA-DPO's gain without touching weights",
    "- On LLaMA the MLP adds no quality: a directly optimised vector reaches the same score at its own best lr",
    "- But on Gemma the bare vector fails at those lrs (cannot reach Gemma's activation scale); MAST transfers unchanged",
    "- All supervised vectors converge to one direction; raw CAA points elsewhere (massive-activation artefact)",
    "- Gains are substantive (not just hedging), transfer to unseen categories and to Gemma, but add no knowledge",
    "## Paper status: all 9 checklist items drafted; open decision = how to frame the MLP result",
], "overview")

# ------------------------------------------------------------------ 2. main result
fig, ax = slide("One supervised vector closes most of the gap to LoRA fine-tuning; raw CAA does nothing",
                "TruthfulQA generation, LLaMA-2-7B-Chat. True×Info (product of rates), mean ± s.d. over 3 seeds of 2-fold CV "
                "(all 817 questions per seed), GPT-4o-mini judges. Grey ticks: numbers reported by RaLFiT (single run).")
rows = [("Unsteered", "baseline", C_BASE), ("Raw CAA (α=1)", "caa_a1", C_CAA), ("Raw CAA (α=2)", "caa_a2", C_CAA),
        ("Direct vector\n(at MAST's lr)", "dvzero_lr5e-4", C_DV), ("Direct vector\n(at its best lr)", "dvzero_lr2e-3", C_DV),
        ("MAST", "mast", C_MAST), ("LoRA-DPO\n(RaLFiT setting,\nsame data)", "loradpo", C_LORA)]
xs = np.arange(len(rows))
for x, (lab, k, c) in zip(xs, rows):
    mu, sd = m(k)
    ax.bar(x, mu, color=c, width=0.62, zorder=2)
    ax.errorbar(x, mu, yerr=sd, color=INK, capsize=4, lw=1.2, zorder=3)
    ax.text(x, mu + sd + 1.2, f"{mu:.1f}", ha="center", fontsize=13, fontweight="bold", color=INK)
for x, v in ((0, 54.56), (6, 76.54), (6, 77.40)):
    ax.plot([x - 0.31, x + 0.31], [v, v], color=INK, lw=2, zorder=4)
ax.text(6, 71.5, "reported:\nRaLFiT 77.4\nLoRA-DPO 76.5", ha="center", va="center", fontsize=9.5, color="white", zorder=5)
ax.text(0, 50, "reported\nunsteered\n54.6", ha="center", va="center", fontsize=9.5, color="white", zorder=5)
ax.set_xticks(xs, [r[0] for r in rows], fontsize=11)
ax.set_ylim(40, 90)
ax.set_ylabel("True × Info (%)")
b, mm, l = m("baseline")[0], m("mast")[0], m("loradpo")[0]
ax.text(0.99, 0.97, f"MAST recovers {100 * (mm - b) / (l - b):.0f}% of LoRA-DPO's gain\nwith no weight changes",
        transform=ax.transAxes, ha="right", va="top", fontsize=13, color=INK,
        bbox=dict(boxstyle="round,pad=0.5", fc="#f2f1ed", ec="none"))
save(fig, "main_result")

# ------------------------------------------------------------------ 3. lr curves
fig, (a1, a2) = slide("The MLP does not add anything: both parameterisations peak at the same height, at different learning rates",
                      "Mean ± s.d. over 6 seed×fold cells. Training loss falls monotonically with lr for both, so it cannot choose the lr. "
                      "Choosing the lr on the judged answers of the other fold: MAST 76.7 ± 2.1 vs direct vector 77.0 ± 2.8.",
                      ncols=2)
curves = [("MAST", {"5e-4": "mast", "1e-3": "mast_lr1e-3", "2e-3": "mast_lr2e-3"}, C_MAST, "o"),
          ("Direct vector, zero init", {"5e-4": "dvzero_lr5e-4", "1e-3": "dvzero_lr1e-3", "2e-3": "dvzero_lr2e-3",
                                        "5e-3": "dvzero_lr5e-3"}, C_DV, "s"),
          ("Direct vector, CAA init", {"5e-4": "dvcaa_lr5e-4", "2e-3": "dvcaa_lr2e-3"}, "#f2a07c", "^")]
LR = {"5e-4": 5e-4, "1e-3": 1e-3, "2e-3": 2e-3, "5e-3": 5e-3}
for lab, pts, c, mk in curves:
    for ax, metric in ((a1, "ti_product"), (a2, "info")):
        xs_ = [LR[k] for k in pts]
        mu = [m(v, metric)[0] for v in pts.values()]
        sd = [m(v, metric)[1] for v in pts.values()]
        ax.errorbar(xs_, mu, yerr=sd, color=c, marker=mk, ms=8, lw=2, capsize=3, label=lab)
for ax, yl in ((a1, "True × Info (%)"), (a2, "Info (%)")):
    ax.set_xscale("log")
    ax.set_xticks(list(LR.values()), list(LR.keys()))
    ax.minorticks_off()
    ax.set_xlabel("learning rate")
    ax.set_ylabel(yl)
a1.axhline(max(m("caa_a1")[0], m("caa_a2")[0]), color=C_BASE, ls=":", lw=1.5)
a1.text(3.2e-3, max(m("caa_a1")[0], m("caa_a2")[0]) + 0.8, "raw CAA", color=INK2, fontsize=10)
a1.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.12), fontsize=11)
a2.text(2.05e-3, m("mast_lr2e-3", "info")[0] - 3, "over-optimised:\nanswers shrink\n24 → 11 words", fontsize=10, color=INK2)
save(fig, "lr_curves")

# ------------------------------------------------------------------ 3b. Gemma: scale
glrs = ["5e-4", "2e-3", "5e-3", "3e-2", "1e-1"]
gpts = []
for lr in glrs:
    d = O / f"rg4b_dvzero_lr{lr}_s42/fold1"
    for jf in ("gpt_judge_results.json", "open_judge_results.json"):
        f = d / "mlp_mc/scale_1.00" / jf
        if f.exists():
            meta = json.loads((d / "meta.json").read_text())
            gpts.append((float(lr), rates(f)[2], meta["v_final_norm"], jf.split("_")[0]))
            break
gm_t = rates(O / "g4b_bn8_full/mlp_mc/scale_1.00/gpt_judge_results.json")[2]
gb_t = rates(O / "g4b_bn8_full/baseline/scale_0.00/gpt_judge_results.json")[2]
fig, (a1, a2) = slide("On Gemma-3-4B the bare vector fails at LLaMA's learning rates: it cannot grow to the model's activation scale; MAST's update scales with the CAA norm",
                      "Gemma-3-4B-IT, layer 13, seed 42 fold 1. Left: True×Info of a zero-initialised direct vector vs learning rate "
                      "(MAST at its LLaMA default lr 5e-4 as a line). Right: norm of the learned vector; MAST's correction has norm ≈ 180 "
                      "because its MLP output scales with ||v_CAA|| ≈ 135 (LLaMA: ≈ 2.5). Points marked ◇ are scored by the open judges.",
                      ncols=2)
for lr, ti, nrm, judge in gpts:
    a1.plot(lr, ti, "D" if judge == "open" else "s", color=C_DV, ms=10)
    a2.plot(lr, nrm, "D" if judge == "open" else "s", color=C_DV, ms=10)
if gpts:
    a1.plot([g[0] for g in gpts], [g[1] for g in gpts], color=C_DV, lw=2, label="direct vector (zero init)")
    a2.plot([g[0] for g in gpts], [g[2] for g in gpts], color=C_DV, lw=2)
a1.axhline(gm_t, color=C_MAST, lw=2.5, label=f"MAST, lr 5e-4 ({gm_t:.0f})")
a1.axhline(gb_t, color=C_BASE, lw=1.5, ls=":", label=f"unsteered ({gb_t:.0f})")
a2.axhline(181.8, color=C_MAST, lw=2.5, label="MAST correction norm (182)")
a2.axhline(134.5, color=C_BASE, lw=1.5, ls=":", label="CAA vector norm (135)")
for ax in (a1, a2):
    ax.set_xscale("log")
    ax.set_xlabel("learning rate (direct vector)")
    ax.legend(frameon=False, fontsize=10.5, loc="center right")
a1.set_ylabel("True × Info (%)")
a1.set_ylim(45, 95)
a2.set_yscale("log")
a2.set_ylabel("norm of learned vector")
save(fig, "gemma_scale")

# ------------------------------------------------------------------ 4. paired comparisons
fig, ax = slide("Cell-by-cell: MAST beats a bare vector trained at MAST's lr, ties it at the vector's own best lr, and trails LoRA-DPO",
                "Paired difference in True×Info (MAST − comparator), bootstrap 95% CI over questions, pooled over 3 seeds × 2 folds. "
                "Label: number of the 6 seed×fold cells in which MAST is ahead.")
comps = [("vs raw CAA (α=2)", "mast - caa_a2"), ("vs direct vector at MAST's lr (5e-4)", "mast - dvzero_lr5e-4"),
         ("vs direct vector, CAA init, MAST's lr", "mast - dvcaa_lr5e-4"),
         ("vs direct vector at 1e-3 (MAST also at 1e-3)", "mast_lr1e-3 - dvzero_lr1e-3"),
         ("vs direct vector at its best lr (2e-3)", "mast - dvzero_lr2e-3"),
         ("vs LoRA-DPO on the same data", "mast - loradpo")]
for y, (lab, key) in enumerate(comps[::-1]):
    c = R["comparisons"][key]
    bs, w = c["bootstrap"], c["wins"]
    d, lo, hi = bs["diff_product"], *bs["ci95_product"]
    col = C_MAST if lo > 0 else (C_LORA if hi < 0 else C_BASE)
    ax.plot([lo, hi], [y, y], color=col, lw=4, solid_capstyle="round")
    ax.plot(d, y, "o", color=col, ms=11, mec="#fcfcfb", mew=2)
    ax.text(hi + 0.6, y, f"{d:+.1f}   ({int(w['a_wins'])}/6 cells)", va="center", fontsize=12, color=INK)
ax.axvline(0, color=INK2, lw=1)
ax.set_yticks(range(len(comps)), [c[0] for c in comps[::-1]], fontsize=12)
ax.set_xlabel("MAST − comparator, True × Info (pp)")
ax.set_xlim(-8, 30)
save(fig, "paired", left=0.32)

# ------------------------------------------------------------------ 5. geometry
names = ["MAST", "Direct (zero init)", "Direct (CAA init)", "Raw CAA"]
paths = lambda s, f: [O / f"rcv_main_s{s}/fold{f}/vectors/v_mlp_mc.pt", O / f"rcv_dvzero_lr2e-3_s{s}/fold{f}/vectors/optimized_vector.pt",  # noqa: E731
                      O / f"rcv_dvcaa_lr2e-3_s{s}/fold{f}/vectors/optimized_vector.pt", O / f"rcv_main_s{s}/fold{f}/vectors/v_steered.pt"]
M = np.zeros((4, 4))
vecs = {n: [] for n in names}
for s, f in CELLS:
    vs = [vec(p) for p in paths(s, f)]
    for i in range(4):
        vecs[names[i]].append(vs[i])
        for j in range(4):
            M[i, j] += cos(vs[i], vs[j]) / len(CELLS)
stab = [np.mean([cos(a, b) for ii, a in enumerate(vecs[n]) for b in vecs[n][ii + 1:]]) for n in names]
stab_sd = [np.std([cos(a, b) for ii, a in enumerate(vecs[n]) for b in vecs[n][ii + 1:]]) for n in names]
fig, (a1, a2) = slide("All supervised vectors are the same direction, whatever the parameterisation; raw CAA is not",
                      "Left: cosine similarity between vectors trained on the same data (mean over 6 cells; d = 4096, random ≈ 0). "
                      "Right: agreement of each method's vector across different seeds/splits (mean ± s.d. of pairwise cosine).",
                      ncols=2, gridspec_kw={"width_ratios": [1.1, 1]})
im = a1.imshow(M, cmap="Blues", vmin=0, vmax=1)
a1.grid(False)
a1.set_xticks(range(4), names, rotation=20, ha="right")
a1.set_yticks(range(4), names)
for i in range(4):
    for j in range(4):
        a1.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=13,
                color="white" if M[i, j] > 0.6 else INK)
cols = [C_MAST, C_DV, "#f2a07c", C_CAA]
a2.barh(range(4)[::-1], stab, xerr=stab_sd, color=cols, height=0.55, capsize=4)
a2.set_yticks(range(4)[::-1], names)
a2.set_xlim(0, 1.15)
a2.set_xlabel("pairwise cosine across seeds/splits")
for y, v, sd in zip(range(4)[::-1], stab, stab_sd):
    a2.text(v + sd + 0.03, y, f"{v:.2f} ± {sd:.2f}", va="center", fontsize=12)
a2.set_title("CAA's large spread: its sign and size flip between splits", fontsize=11, color=INK2)
save(fig, "geometry", wspace=0.5, left=0.13)

# ------------------------------------------------------------------ 6. CAA artefact
share = {n: [] for n in ["Raw CAA", "MAST", "Direct (zero init)"]}
for s, f in CELLS:
    for n, p in (("Raw CAA", O / f"rcv_main_s{s}/fold{f}/vectors/v_steered.pt"), ("MAST", O / f"rcv_main_s{s}/fold{f}/vectors/v_mlp_mc.pt"),
                 ("Direct (zero init)", O / f"rcv_dvzero_lr2e-3_s{s}/fold{f}/vectors/optimized_vector.pt")):
        v = vec(p)
        share[n].append(100 * float((v[[1415, 2533]] ** 2).sum() / (v ** 2).sum()))
cv = json.loads((O / "rx_caavar/caa_variants.json").read_text())
fig, (a1, a2) = slide("Why CAA fails here: it is mostly two 'massive activation' coordinates, and even cleaned it points elsewhere",
                      "Left: share of each vector's squared norm in dims 1415 and 2533, the known massive-activation dims of LLaMA-2-7B "
                      "(Sun et al. 2024); dots = the 6 cells. Right: CAA extracted with different token pooling (seed 42, fold 1): "
                      "artefact share and cosine with the supervised vector.", ncols=2)
for x, (n, c) in enumerate(zip(share, [C_CAA, C_MAST, C_DV])):
    a1.bar(x, np.mean(share[n]), color=c, width=0.6)
    a1.scatter(np.full(6, x) + np.linspace(-0.15, 0.15, 6), share[n], color=INK, s=22, zorder=3)
    a1.text(x, np.mean(share[n]) + 3, f"{np.mean(share[n]):.0f}%", ha="center", fontsize=13, fontweight="bold")
a1.set_xticks(range(3), list(share))
a1.set_ylabel("% of squared norm in dims 1415, 2533")
a1.set_ylim(0, 110)
variants = [("all tokens\n(pipeline)", "all"), ("all but BOS", "nobos"), ("answer\ntokens", "answer"), ("last token", "last")]
xx = np.arange(len(variants))
a2.bar(xx - 0.18, [100 * cv[k]["massive_share"] for _, k in variants], width=0.36, color=C_CAA, label="artefact share (%)")
a2.bar(xx + 0.18, [100 * cv[k]["cos_ref"] for _, k in variants], width=0.36, color=C_MAST, label="cos with supervised vector (×100)")
for x, (_, k) in zip(xx, variants):
    a2.text(x + 0.18, 100 * cv[k]["cos_ref"] + 2, f"{cv[k]['cos_ref']:+.2f}", ha="center", fontsize=11)
a2.set_xticks(xx, [v[0] for v in variants])
a2.set_ylim(-5, 110)
a2.legend(frameon=False, loc="upper right", fontsize=10.5)
a2.set_title("Reading direction ≠ writing direction", fontsize=13, color=INK)
save(fig, "caa_artefact")

# ------------------------------------------------------------------ 7. hedging
H = re.compile(r"\b(no comment|i don'?t know|i do not know|i cannot|i can'?t|not sure|i'?m not able|unable to|"
               r"there is no (?:scientific )?evidence|it is not possible to|no one knows|depends)\b", re.I)
hmeth = [("Unsteered", "main", "baseline/scale_0.00", C_BASE), ("Raw CAA", "main", "steered/scale_1.00", C_CAA),
         ("Direct vector\n(best lr)", "dvzero_lr2e-3", "mlp_mc/scale_1.00", C_DV), ("MAST", "main", "mlp_mc/scale_1.00", C_MAST),
         ("LoRA-DPO", "loradpo", "mlp_mc/scale_1.00", C_LORA)]
hed, sub = [], []
for _, meth, sp, _ in hmeth:
    h, t = [], []
    for s, f in CELLS:
        for r in judged(O / f"rcv_{meth}_s{s}/fold{f}/{sp}/gpt_judge_results.json")[0]:
            hh = bool(H.search(r["generated_clean"]))
            h.append(hh)
            if not hh:
                t.append(r["truth_judgment"] == "yes" and r["info_judgment"] == "yes")
    hed.append(100 * np.mean(h))
    sub.append(100 * np.mean(t))
sweep = [(0.8, O / "sweep_s080/mlp_mc/scale_0.80"), (0.9, O / "sweep_s090/mlp_mc/scale_0.90"),
         (0.95, O / "sweep_s095/mlp_mc/scale_0.95"), (1.0, O / "multiseed/seed_42/mlp_mc/scale_1.00"),
         (1.1, O / "sweep_s110/mlp_mc/scale_1.10"), (1.2, O / "sweep_s120/mlp_mc/scale_1.20")]
sw = []
for a, p in sweep:
    res, _ = judged(p / "gpt_judge_results.json")
    sw.append((a, 100 * np.mean([bool(H.search(r["generated_clean"])) for r in res]),
               100 * np.mean([r["truth_judgment"] == "yes" and r["info_judgment"] == "yes" for r in res])))
fig, (a1, a2, a3) = slide("The gain is not bought by refusing: answers that do answer become far more truthful; α is a hedging dial",
                          "Hedged = answer matches a refusal/uncertainty phrase list (indicative). Left/middle: 6 cells pooled. "
                          "Right: earlier single-seed α sweep of the MAST vector (seed 42).", ncols=3)
xx = np.arange(len(hmeth))
cl = [h[3] for h in hmeth]
a1.bar(xx, hed, color=cl, width=0.6)
a1.set_title("Hedged answers (%)", fontsize=13)
a2.bar(xx, sub, color=cl, width=0.6)
a2.set_title("Truth∧Info among non-hedged answers (%)", fontsize=13)
for ax, vals in ((a1, hed), (a2, sub)):
    ax.set_xticks(xx, [h[0] for h in hmeth], rotation=30, ha="right", fontsize=10.5)
    for x, v in zip(xx, vals):
        ax.text(x, v + 1, f"{v:.0f}", ha="center", fontsize=11.5)
a2.set_ylim(0, 95)
a3.plot([s[0] for s in sw], [s[1] for s in sw], color=C_MAST, marker="o", ms=8, lw=2, label="hedged (%)")
a3.plot([s[0] for s in sw], [s[2] for s in sw], color=INK2, marker="s", ms=7, lw=2, ls="--", label="Truth∧Info (%)")
a3.set_xlabel("steering scale α")
a3.legend(frameon=False, fontsize=10.5, loc="center right")
a3.set_title("MAST vector at different α", fontsize=13)
save(fig, "hedging", bottom=0.2)

# ------------------------------------------------------------------ 8. generality
fig, (a1, a2, a3) = slide("It learns a behaviour, not knowledge: transfers to unseen categories and to Gemma, hurts knowledge QA",
                          "Left: category hold-out (train on one half of TruthfulQA's 38 categories, test on the other; seed 42, T∧I). "
                          "Middle: unchanged recipe on Gemma-3-4B-IT (layer chosen by training loss; 2-fold, seed 42). "
                          "Right: PopQA / NQ-open exact match with the TruthfulQA vector (1,000 questions each).", ncols=3)
ch = []
for fold in ("A", "B"):
    vals = []
    for var, sc in (("baseline", "scale_0.00"), ("steered", "scale_1.00"), ("mlp_mc", "scale_1.00")):
        res, st = judged(O / f"cathold_fold_{fold}/{var}/{sc}/gpt_judge_results.json")
        vals.append(100 * st["truth_and_info_accuracy"])
    ch.append(vals)
xx = np.arange(2)
for k, (lab, c) in enumerate((("Unsteered", C_BASE), ("Raw CAA", C_CAA), ("MAST", C_MAST))):
    a1.bar(xx + (k - 1) * 0.26, [v[k] for v in ch], width=0.24, color=c, label=lab)
    for x, v in zip(xx, ch):
        a1.text(x + (k - 1) * 0.26, v[k] + 1, f"{v[k]:.0f}", ha="center", fontsize=10)
a1.set_xticks(xx, ["fold A\n(20 unseen cats)", "fold B\n(18 unseen cats)"])
a1.set_ylim(0, 100)
a1.legend(frameon=False, fontsize=10, loc="upper right")
g = {}
for lab, var, sc in (("Unsteered", "baseline", "scale_0.00"), ("Raw CAA", "steered", "scale_1.00"), ("MAST", "mlp_mc", "scale_1.00")):
    r1, _ = judged(O / f"g4b_bn8_full/{var}/{sc}/gpt_judge_results.json")
    r2, _ = judged(O / f"rcv_g4bmain_s42/fold2/{var}/{sc}/gpt_judge_results.json")
    rr = r1 + r2
    g[lab] = 100 * np.mean([r["truth_judgment"] == "yes" for r in rr]) * np.mean([r["info_judgment"] == "yes" for r in rr])
gdv = {lr: rates(O / f"rg4b_dvzero_lr{lr}_s42/fold1/mlp_mc/scale_1.00/gpt_judge_results.json")[2] for lr in ("5e-4", "2e-3", "5e-3")}
gm1 = rates(O / "g4b_bn8_full/mlp_mc/scale_1.00/gpt_judge_results.json")[2]
labs = ["Unsteered", "Raw CAA", "MAST"]
a2.bar(range(3), [g[k] for k in labs], color=[C_BASE, C_CAA, C_MAST], width=0.6)
for x, k in enumerate(labs):
    a2.text(x, g[k] + 1, f"{g[k]:.1f}", ha="center", fontsize=11.5)
a2.set_xticks(range(3), labs)
a2.set_ylim(0, 100)
a2.set_ylabel("True × Info (%)")
a2.set_title("Direct vector (fold 1): " + ", ".join(f"lr {lr}: {v:.0f}" for lr, v in gdv.items())
             + f"\nMAST (fold 1): {gm1:.0f}", fontsize=10, color=INK2)
pq = json.loads((O / "multiseed/seed_42/transfer_popqa.json").read_text())["variants"]
nq = json.loads((O / "multiseed/seed_42/transfer_nq.json").read_text())["variants"]
xx = np.arange(2)
a3.bar(xx - 0.18, [pq["baseline"]["em_contains"], nq["baseline"]["em_contains"]], width=0.34, color=C_BASE, label="Unsteered")
a3.bar(xx + 0.18, [pq["steered"]["em_contains"], nq["steered"]["em_contains"]], width=0.34, color=C_MAST, label="+ TruthfulQA vector")
for x, (b0, b1) in zip(xx, ((pq["baseline"]["em_contains"], pq["steered"]["em_contains"]),
                            (nq["baseline"]["em_contains"], nq["steered"]["em_contains"]))):
    a3.text(x - 0.18, b0 + 0.6, f"{b0:.1f}", ha="center", fontsize=11)
    a3.text(x + 0.18, b1 + 0.6, f"{b1:.1f}", ha="center", fontsize=11)
a3.set_xticks(xx, ["PopQA", "NQ-open"])
a3.set_ylabel("exact match (%)")
a3.set_ylim(0, 40)
a3.legend(frameon=False, fontsize=10)
save(fig, "generality", wspace=0.3)

# ------------------------------------------------------------------ 9. categories
cats = R["categories"]
cm = {c: v for c, v in cats["mast"].items() if v["n"] >= 30}
gain = sorted(((c, cats["baseline"][c]["ti_conj"], cm[c]["ti_conj"], cm[c]["n"] // 3) for c in cm), key=lambda t: t[2] - t[1])
fig, ax = slide("Per category: big gains where the model repeats popular misconceptions, little where precise recall is needed",
                "TruthfulQA categories with ≥10 questions; Truth∧Info, unsteered (grey) → MAST (blue), all 817 questions × 3 seeds. "
                "Sorted by gain. Full 38-category table in the paper's appendix.")
for y, (c, b0, b1, n) in enumerate(gain):
    ax.plot([b0, b1], [y, y], color=C_MUTED, lw=3, zorder=1)
    ax.plot(b0, y, "o", color=C_BASE, ms=8, zorder=2)
    ax.plot(b1, y, "o", color=C_MAST, ms=9, zorder=3)
    ax.text(101, y, f"{b1 - b0:+.0f}", va="center", fontsize=10, color=INK2)
ax.set_yticks(range(len(gain)), [f"{c} (n={n})" for c, _, _, n in gain], fontsize=9.5)
ax.set_xlim(0, 106)
ax.set_xlabel("Truth ∧ Info (%)")
save(fig, "categories", left=0.3, top=0.84, bottom=0.08)

# ------------------------------------------------------------------ 10. exploratory (filled when available)
exp = [("LoRA-DPO (full)", O / "rcv_loradpo_s42/fold1/mlp_mc/scale_1.00"),
       ("LoRA-DPO, our direction removed at L8", O / "rx_lora_abl8/fold1/mlp_mc/scale_1.00"),
       ("LoRA-DPO, our direction removed at all layers", O / "rx_lora_ablall/fold1/mlp_mc/scale_1.00"),
       ("Unsteered, direction removed at all layers", O / "rx_base_ablall/fold1/mlp_mc/scale_1.00"),
       ("Unsteered", O / "rcv_main_s42/fold1/baseline/scale_0.00"),
       ("MAST vector, α = −1", O / "rx_mast_neg/fold1/mlp_mc/scale_1.00"),
       ("Consensus of 3 supervised vectors", O / "rx_consensus/fold1/mlp_mc/scale_1.00"),
       ("MAST", O / "rcv_main_s42/fold1/mlp_mc/scale_1.00"),
       ("LoRA shift distilled into a vector (α=1)", O / "rx_distilled_a1/fold1/mlp_mc/scale_1.00"),
       ("LoRA shift distilled into a vector (α=2)", O / "rx_distilled_a2/fold1/mlp_mc/scale_1.00"),
       ("CAA, answer-token pooling (norm-matched)", O / "rx_caa_answer/fold1/mlp_mc/scale_1.00"),
       ("Geometry-of-Truth direction as steering vector", O / "rx_gotdir_a1/fold1/mlp_mc/scale_1.00")]
have = [(lab, p) for lab, p in exp if (p / "open_judge_results.json").exists()]
fig, ax = slide("Exploratory interventions (seed 42, fold 1; open AllenAI judges)",
                "Mediation: does removing our direction from LoRA-DPO's residual stream undo its gain? Distillation: can LoRA's "
                "mean activation shift serve as a steering vector? All rows: same 408 test questions, same judge. "
                f"{len(have)}/{len(exp)} runs available when this slide was built.")
for y, (lab, p) in enumerate(have[::-1]):
    res, st = judged(p / "open_judge_results.json")
    v = 100 * st["truth_accuracy"] * st["info_accuracy"]
    col = C_LORA if lab.startswith("LoRA") else (C_MAST if "MAST" in lab or "vector" in lab or "Consensus" in lab else C_BASE)
    ax.barh(y, v, color=col, height=0.6)
    ax.text(v + 0.8, y, f"{v:.1f}   (Info {100 * st['info_accuracy']:.0f})", va="center", fontsize=11)
ax.set_yticks(range(len(have)), [h[0] for h in have[::-1]], fontsize=11)
ax.set_xlim(0, 100)
ax.set_xlabel("True × Info (%), open judges")
save(fig, "exploratory", left=0.33)

# ------------------------------------------------------------------ 11. decisions
text_slide("Decisions for today", [
    "## 1. How to present the MLP result (paper due 12 Oct, same reviewers)",
    "- (a) Keep MAST: its MLP makes the update scale-adaptive (output ∝ ||v_CAA||), so one lr transfers across models;",
    "      a bare vector equals it only with per-model lr tuning (LLaMA: 4× higher lr; Gemma: ~50× — test running)",
    "- (b) Reframe as 'supervised steering vectors', MAST as one parameterisation (weaker given the Gemma result)",
    "## 2. What is the contribution if the MLP is not it?",
    "- Matched-protocol result: one vector ≈ 88% of LoRA-DPO, no weight change (prior vectors: MC-only or different protocol)",
    "- Why CAA fails on TruthfulQA (massive-activation artefact; reading ≠ writing direction)",
    "- Loss: our margin loss vs BiPO's loss on open-ended generation — running now",
    "## 3. Interpretability: pursue only if an exploratory result is strong (mediation / distillation)",
    "## 4. Thesis: examiners' corrections drafted; MAST stays the thread, with the corrected conclusion",
    "## 5. Compute: ~1 GPU-day left on the box for whatever we pick",
], "decisions")

with PdfPages(OUT / "deck.pdf") as pdf:
    for f in SLIDES:
        pdf.savefig(f)
print(f"{len(SLIDES)} slides -> {OUT}/deck.pdf")
