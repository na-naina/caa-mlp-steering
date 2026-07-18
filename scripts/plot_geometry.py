"""Plot geometry analysis of trained MAST correction: active neurons and cosine to v_CAA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--k8-json", type=Path, required=True,
                   help="JSON with 10-seed-ish k=8 geometry results")
    p.add_argument("--k16-json", type=Path, required=True,
                   help="JSON with k=16 geometry results")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    with args.k8_json.open() as f:
        k8 = json.load(f)
    with args.k16_json.open() as f:
        k16 = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))

    # Panel 1: k=16 active neurons
    k16_entry = k16[0]
    k16_acts = np.array(k16_entry["post_relu_values"])
    colors16 = ["#1f77b4" if v > 0 else "#cccccc" for v in k16_acts]
    axes[0].bar(range(len(k16_acts)), k16_acts, color=colors16, edgecolor="black", linewidth=0.3)
    axes[0].set_xlabel("Bottleneck neuron index")
    axes[0].set_ylabel("Post-ReLU activation")
    active16 = int((k16_acts > 1e-6).sum())
    axes[0].set_title(f"k=16 (seed {k16_entry['seed']}): {active16} active / 16")
    axes[0].set_xticks(range(len(k16_acts)))
    axes[0].tick_params(axis="x", labelsize=7)

    # Panel 2: k=8 activation patterns across seeds (stacked)
    seeds = [e["seed"] for e in k8]
    acts = np.array([e["post_relu_values"] for e in k8])  # [n_seeds, 8]
    im = axes[1].imshow(acts, cmap="viridis", aspect="auto")
    axes[1].set_yticks(range(len(seeds)))
    axes[1].set_yticklabels([f"seed {s}" for s in seeds])
    axes[1].set_xticks(range(8))
    axes[1].set_xlabel("Bottleneck neuron index")
    axes[1].set_title("k=8: firing pattern per seed")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="post-ReLU")

    # Panel 3: cosine similarity scatter
    cos_k8 = [e["cosine_delta_vcaa"] for e in k8]
    cos_k16 = [k16_entry["cosine_delta_vcaa"]]
    axes[2].scatter([8] * len(cos_k8), cos_k8, s=60, alpha=0.7, label="k=8", color="#1f77b4")
    axes[2].scatter([16] * len(cos_k16), cos_k16, s=60, alpha=0.7, label="k=16", color="#ff7f0e")
    axes[2].axhline(0.0, color="gray", linewidth=0.5, linestyle="--")
    axes[2].set_xlim(4, 20)
    axes[2].set_xticks([8, 16])
    axes[2].set_ylim(-0.01, max(cos_k8 + cos_k16) * 1.6)
    axes[2].set_xlabel("Bottleneck dimension k")
    axes[2].set_ylabel(r"$\cos(\delta, v_{CAA})$")
    axes[2].set_title("Correction $\\delta$ vs CAA vector")
    axes[2].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=300)
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
