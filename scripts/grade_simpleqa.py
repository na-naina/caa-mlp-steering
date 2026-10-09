#!/usr/bin/env python3
"""Grade scripts/eval_simpleqa.py generations with the SimpleQA reference-based autorater (runs locally).

Grader: OpenAI's SimpleQA grader prompt (src/prompts/simpleqa_grader.py) with GPT-4.1, the autorater model of
SimpleQA Verified; --template verified switches to the Verified prompt once it has been pasted in.
Reports, per variant, the standard SimpleQA metrics: correct %, incorrect %, not-attempted %,
correct-given-attempted %, and F-score (harmonic mean of correct and correct-given-attempted).

    python scripts/grade_simpleqa.py data/outputs/sqv_*_s42f1      # writes <dir>/grades.json + summary
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.prompts.simpleqa_grader import GRADER_TEMPLATE, SIMPLEQA_VERIFIED_TEMPLATE  # noqa: E402

load_dotenv()
LETTER = {"A": "correct", "B": "incorrect", "C": "not_attempted"}


def grade_one(client, model, template, q, target, pred):
    for _ in range(5):
        try:
            r = client.chat.completions.create(
                model=model, temperature=0, max_tokens=4,
                messages=[{"role": "user", "content": template.format(
                    question=q, target=target, predicted_answer=pred)}])
            m = re.search(r"(A|B|C)", r.choices[0].message.content or "")
            return m.group(0) if m else "C"
        except Exception as e:  # rate limits / transient errors
            err = e
    raise err


def summarise(letters):
    n = len(letters)
    c, i, na = (100 * sum(x == k for x in letters) / n for k in "ABC")
    cga = 100 * c / (c + i) if c + i else 0.0
    f = 2 * c * cga / (c + cga) if c + cga else 0.0
    return {"n": n, "correct": c, "incorrect": i, "not_attempted": na,
            "correct_given_attempted": cga, "f_score": f}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--grader", default="gpt-4.1")
    ap.add_argument("--template", choices=["openai", "verified"], default="openai")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    template = GRADER_TEMPLATE if a.template == "openai" else SIMPLEQA_VERIFIED_TEMPLATE
    if template is None:
        sys.exit("SIMPLEQA_VERIFIED_TEMPLATE is not set in src/prompts/simpleqa_grader.py")
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    for d in a.dirs:
        g = json.loads((d / "generations.json").read_text())
        dest = d / "grades.json"
        out = json.loads(dest.read_text()) if dest.exists() else {"grader": a.grader, "template": a.template,
                                                                   "letters": {}, "summary": {}}
        items = g["items"]
        for variant, preds in g["preds"].items():
            if variant in out["letters"]:
                continue
            with ThreadPoolExecutor(a.workers) as ex:
                letters = list(ex.map(lambda ip: grade_one(client, a.grader, template, ip[0]["question"],
                                                           ip[0]["target"], ip[1]), zip(items, preds)))
            out["letters"][variant] = letters
            out["summary"][variant] = summarise(letters)
            dest.write_text(json.dumps(out, indent=1))
        print(f"\n{d}  ({g['model']}, layer {g['layer']}, grader {a.grader}/{a.template})")
        print(f"{'variant':10s} {'correct':>8s} {'incorrect':>9s} {'not_att':>8s} {'CgA':>6s} {'F':>6s}")
        for v, s in out["summary"].items():
            print(f"{v:10s} {s['correct']:8.1f} {s['incorrect']:9.1f} {s['not_attempted']:8.1f} "
                  f"{s['correct_given_attempted']:6.1f} {s['f_score']:6.1f}")


if __name__ == "__main__":
    main()
