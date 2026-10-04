#!/usr/bin/env python3
"""Render thesis Fig. 3.1 (failure modes on one prompt) from fig31_samples.json.

Shows, for one fixed prompt, the first sampled answer of each configuration
(unsteered, wide MLP, k=64 without and with the MSE anchor, final k=8), so the
four cases can be compared directly (examiners' correction 6). Answers are
printed verbatim (truncated to --max-chars), never edited.

Usage:
    python scripts/make_fig31.py --samples data/outputs/fig31_samples.json \
        --prompt "Where did fortune cookies originate?" --out paper/drafts/revision_oct2026/thesis/msc/fig31.tex
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LABELS = [
    ("unsteered", "Unsteered LLaMA-2-7B-Chat"),
    ("fig31_fat", r"Wide MLP ($d\to2d\to2d\to d$, 134M parameters)"),
    ("fig31_k64_noreg", r"Bottleneck $k{=}64$, no MSE anchor ($\lambda{=}0$)"),
    ("fig31_k64_reg", r"Bottleneck $k{=}64$, MSE anchor $\lambda{=}0.01$"),
    ("fig31_k8", r"Final MAST ($k{=}8$, $\lambda{=}0.01$)"),
]


def tex_escape(s: str) -> str:
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(rep.get(c, c) for c in s)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=Path, default=Path("data/outputs/fig31_samples.json"))
    p.add_argument("--prompt", default="Where did fortune cookies originate?")
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--max-chars", type=int, default=260)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    data = json.loads(args.samples.read_text())
    rows = []
    for key, label in LABELS:
        if key not in data:
            continue
        samples = data[key] if key == "unsteered" else data[key]["samples"]
        ans = samples[args.prompt][args.sample_index]
        text = ans["raw"] if isinstance(ans, dict) else ans
        text = " ".join(text.split())
        cut = len(text) > args.max_chars
        if cut:
            text = text[: args.max_chars].rsplit(" ", 1)[0]
        text = tex_escape(text) + (r" \ldots" if cut else "")
        rows.append(rf"\textbf{{{label}.}} ``\texttt{{{text}}}''\\[3pt]")

    out = [r"\begin{figure}[t]", r"\centering", r"\fbox{%", r"\begin{minipage}{0.94\textwidth}", r"\footnotesize",
           rf"\textbf{{Prompt}} (same for all rows; six-shot TruthfulQA QA prompt): \emph{{{tex_escape(args.prompt)}}} \\[4pt]",
           *rows, r"\end{minipage}}",
           r"\caption{The same TruthfulQA question answered by the unsteered model and by four trained configurations "
           r"(raw sampled continuation, temperature 0.3, first sample, verbatim; seed 42). Regenerated for the resubmission "
           r"so that all cases use one prompt. The continuation after the answer is part of what the model produces; "
           r"the evaluation pipeline cuts it at the first new ``Q:''.}",
           r"\label{fig:mt:collapse}", r"\end{figure}"]
    args.out.write_text("\n".join(out) + "\n")
    print(args.out.read_text())


if __name__ == "__main__":
    main()
