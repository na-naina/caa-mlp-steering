#!/usr/bin/env python3
"""Held-out A/B eval for a behavior-trained steering vector.

For each held-out item, compares the steered model's mean log-prob of the
behavior-matching vs non-matching option at scales {-1, 0, +1} and reports
the behavior-matching choice rate. CAA convention: the trained vector points
TOWARD the behavior; scale -1 is the mitigation direction.

Usage:
    python scripts/eval_behavior_ab.py --run-dir data/outputs/halluc_vector \
        --behavior hallucination
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.loader import load_causal_model
from src.steering.apply import steering_hook
from src.data.behavior_ab import BehaviorABDataset
from src.utils.batching import build_prompt_answer_batch
from src.utils.scoring import compute_answer_logprobs

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
LOG = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--behavior", required=True)
    p.add_argument("--vector-file", default="vectors/v_mast_mc.pt")
    p.add_argument("--scales", nargs="+", type=float, default=[-1.0, 0.0, 1.0])
    args = p.parse_args()

    cfg = yaml.safe_load((args.run_dir / "config.yaml").read_text())
    model_name, layer = cfg["model"]["name"], cfg["model"]["layer"]

    ds = BehaviorABDataset(behavior=args.behavior, seed=42)
    held = ds.heldout_ab_items()
    LOG.info("%s: %d held-out A/B items | %s L%d", args.behavior, len(held), model_name, layer)

    loaded = load_causal_model(model_name, dtype="bfloat16", device_map="auto")
    model, tok, dev = loaded.model, loaded.tokenizer, loaded.primary_device
    model.eval()
    vector = torch.load(args.run_dir / args.vector_file, map_location="cpu")

    prompts = [f"Question: {it['question']}\nAnswer:" for it in held]
    # dataset convention: best_answer = behavior-MATCHING option; incorrect_answers[0] = other
    match_ans = [it["best_answer"] for it in held]
    non_ans = [it["incorrect_answers"][0] for it in held]

    out = {"behavior": args.behavior, "n": len(held), "scales": {}}
    for s in args.scales:
        ctx = steering_hook(model, layer, vector, scale=s) if s != 0 else None
        with torch.no_grad():
            cm = ctx.__enter__() if ctx else None
            try:
                im = build_prompt_answer_batch(tok, prompts, match_ans, max_length=512)
                inm = build_prompt_answer_batch(tok, prompts, non_ans, max_length=512)
                lp_m, _ = compute_answer_logprobs(model, input_ids=im[0].to(dev),
                                                  attention_mask=im[1].to(dev), answer_mask=im[2].to(dev))
                lp_n, _ = compute_answer_logprobs(model, input_ids=inm[0].to(dev),
                                                  attention_mask=inm[1].to(dev), answer_mask=inm[2].to(dev))
            finally:
                if ctx:
                    ctx.__exit__(None, None, None)
        rate = (lp_m > lp_n).float().mean().item()
        out["scales"][str(s)] = {"behavior_match_rate": 100 * rate}
        LOG.info("scale %+0.1f: behavior-matching choice %.1f%%", s, 100 * rate)

    (args.run_dir / f"ab_eval_{args.behavior}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
