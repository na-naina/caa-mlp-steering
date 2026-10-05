#!/usr/bin/env python3
"""Exploratory interventions on TruthfulQA generation (interpretability follow-ups).

One configurable run = one set of test answers, written to
  <out>/fold1/mlp_mc/scale_1.00/generation_details.json
(the layout both judge scripts discover). Interventions compose:

  --lora PATH               merge a LoRA adapter into the model first
  --add VEC[:SCALE] ...     add vector(s) at --layer (VEC is a .pt file)
  --ablate VEC              project VEC's direction out of the residual stream
  --ablate-layers 8|all     where to ablate (output of those blocks)
  --distill-lora PATH       save the mean layer-L residual shift (LoRA minus base) over
                            training-half prompts as a steering vector, then exit

Usage examples:
  # Is LoRA-DPO's gain mediated by the supervised direction?
  python scripts/steer_explore.py --splits-file data/splits/cv2_s42_fold1.json \
      --lora data/outputs/rcv_loradpo_s42/fold1/lora_adapter \
      --ablate data/outputs/rcv_main_s42/fold1/vectors/v_mlp_mc.pt --ablate-layers all \
      --out data/outputs/rx_lora_ablall
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.truthfulqa import TruthfulQADatasetManager, TruthfulQAPipelineSplits  # noqa: E402
from src.evaluation.truthfulqa import evaluate_generation  # noqa: E402
from src.models.loader import load_causal_model  # noqa: E402
from src.prompts.truthfulqa_presets import format_prompt  # noqa: E402
from src.steering.apply import _get_decoder_layer  # noqa: E402

GEN_CFG = {"preset": "qa", "temperature": 0.3, "top_p": 0.9, "max_new_tokens": 64,
           "max_length": 512, "stop_sequences": ["\n\n", "\nQuestion:"]}


def ablation_hook(direction):
    def hook(_m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        d = direction.to(h.device, dtype=h.dtype)
        d = d / d.norm()
        h = h - (h * d).sum(-1, keepdim=True) * d
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
    return hook


def add_hook(vector):
    def hook(_m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h + vector.to(h.device, dtype=h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
    return hook


@torch.no_grad()
def mean_resid(model, tok, prompts, layer, batch=16):
    """Mean over all non-pad positions of the layer-`layer` block output."""
    store = {}
    hd = _get_decoder_layer(model, layer).register_forward_hook(
        lambda m, i, o: store.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
    tot, n = None, 0
    tok.padding_side = "right"
    for s in range(0, len(prompts), batch):
        enc = tok(prompts[s:s + batch], return_tensors="pt", padding=True).to(model.device)
        model(**enc)
        m = enc["attention_mask"].unsqueeze(-1).to(store["h"].dtype)
        sm = (store["h"].float() * m.float()).sum((0, 1))
        tot = sm if tot is None else tot + sm
        n += int(m.sum())
    hd.remove()
    return (tot / n).cpu()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
    p.add_argument("--layer", type=int, default=8)
    p.add_argument("--splits-file", type=Path, required=True)
    p.add_argument("--lora", type=Path)
    p.add_argument("--add", nargs="*", default=[])
    p.add_argument("--ablate", type=Path)
    p.add_argument("--ablate-layers", default="8")
    p.add_argument("--distill-lora", type=Path, help="adapter to distil into a vector (writes --out .pt)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    torch.manual_seed(args.seed)

    sd = json.loads(args.splits_file.read_text())
    splits = TruthfulQAPipelineSplits(steering_pool=sd["steering_pool"], train=sd["train"], test=sd["test"],
                                      val=sd.get("val", []))
    dataset = TruthfulQADatasetManager(seed=args.seed)
    loaded = load_causal_model(args.model, dtype="bfloat16", device_map="auto")
    model, tok, dev = loaded.model, loaded.tokenizer, loaded.primary_device
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()

    if args.distill_lora:
        prompts = [format_prompt(dataset.get_item(int(i))["question"], preset="qa")
                   for i in list(splits.steering_pool) + list(splits.train)]
        base_mean = mean_resid(model, tok, prompts, args.layer)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(args.distill_lora)).merge_and_unload().eval()
        lora_mean = mean_resid(model, tok, prompts, args.layer)
        delta = lora_mean - base_mean
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(delta, args.out)
        print(f"distilled vector: norm {delta.norm():.3f} -> {args.out}")
        return

    if args.lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(args.lora)).merge_and_unload().eval()

    handles = []
    for spec in args.add:
        path, _, scale = spec.partition(":")
        v = torch.load(path, map_location="cpu").float().flatten() * (float(scale) if scale else 1.0)
        handles.append(_get_decoder_layer(model, args.layer).register_forward_hook(add_hook(v)))
    if args.ablate:
        d = torch.load(args.ablate, map_location="cpu").float().flatten()
        layers = range(model.config.num_hidden_layers) if args.ablate_layers == "all" else \
            [int(x) for x in args.ablate_layers.split(",")]
        for li in layers:
            handles.append(_get_decoder_layer(model, li).register_forward_hook(ablation_hook(d)))

    result = evaluate_generation(model, tok, dataset.get_items(splits.test), layer_index=args.layer,
                                 steering_vector=None, scale=0.0, generation_cfg=GEN_CFG,
                                 primary_device=dev, judge=None, semantic_judge=None)
    for h in handles:
        h.remove()
    out = args.out / "fold1" / "mlp_mc" / "scale_1.00"
    out.mkdir(parents=True, exist_ok=True)
    (out / "generation_details.json").write_text(json.dumps(result["details"], indent=2))
    (args.out / "fold1" / "meta.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    print(f"{len(result['details'])} answers -> {out}")


if __name__ == "__main__":
    main()
