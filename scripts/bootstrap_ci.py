#!/usr/bin/env python3
"""Bootstrap confidence intervals for TruthfulQA GPT judge results."""

import argparse
import json
import numpy as np
from pathlib import Path


def load_judgments(json_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load per-example truth/info judgments as boolean arrays."""
    with open(json_path) as f:
        data = json.load(f)
    results = data["results"]
    truth = np.array([r["truth_judgment"] == "yes" for r in results])
    info = np.array([r["info_judgment"] == "yes" for r in results])
    return truth, info


def bootstrap_ci(truth: np.ndarray, info: np.ndarray,
                 n_boot: int = 10000, ci: float = 0.95,
                 seed: int = 42) -> dict:
    """Compute bootstrap CIs for Truth%, Info%, T*I%."""
    rng = np.random.RandomState(seed)
    n = len(truth)
    lo = (1 - ci) / 2
    hi = 1 - lo

    truth_boots = np.empty(n_boot)
    info_boots = np.empty(n_boot)
    conj_boots = np.empty(n_boot)  # per-item conjunction: mean(T_i & I_i)
    prod_boots = np.empty(n_boot)  # product of means: mean(T) * mean(I)

    conj = truth & info
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        t = truth[idx].mean()
        inf = info[idx].mean()
        truth_boots[i] = t
        info_boots[i] = inf
        conj_boots[i] = conj[idx].mean()
        prod_boots[i] = t * inf

    def _stats(point, boots):
        return {
            "mean": point * 100,
            "ci_lo": np.percentile(boots, lo * 100) * 100,
            "ci_hi": np.percentile(boots, hi * 100) * 100,
        }

    return {
        "truth": _stats(truth.mean(), truth_boots),
        "info": _stats(info.mean(), info_boots),
        # Per-item conjunction (the repo's historical "T*I"): P(truthful AND informative).
        "t_and_i": _stats(conj.mean(), conj_boots),
        # Product of means, as reported by RaLFiT/prior work: Truth% x Info%.
        "t_times_i": _stats(truth.mean() * info.mean(), prod_boots),
    }


def find_result_files(results_dir: Path, variants: list[str] | None = None):
    """Find gpt_judge_results.json files for each variant."""
    if variants is None:
        variants = ["baseline", "steered", "mlp_mc", "mlp_gen"]

    found = {}
    for variant in variants:
        # Try direct: <dir>/<variant>/scale_*/gpt_judge_results.json
        variant_dir = results_dir / variant
        if variant_dir.is_dir():
            for scale_dir in sorted(variant_dir.iterdir()):
                jf = scale_dir / "gpt_judge_results.json"
                if jf.exists():
                    found[f"{variant}/{scale_dir.name}"] = jf
                    break  # take first scale
        # Try fold structure: <dir>/fold*/variant/scale_*/
        for fold_dir in sorted(results_dir.glob("fold*")):
            vd = fold_dir / variant
            if vd.is_dir():
                for scale_dir in sorted(vd.iterdir()):
                    jf = scale_dir / "gpt_judge_results.json"
                    if jf.exists():
                        found[f"{fold_dir.name}/{variant}/{scale_dir.name}"] = jf
                        break
    return found


def main():
    parser = argparse.ArgumentParser(description="Bootstrap CIs for TruthfulQA results")
    parser.add_argument("dirs", nargs="+", help="Result directories to analyze")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--json-out", type=str, default=None, help="Save results as JSON")
    args = parser.parse_args()

    all_results = {}

    for d in args.dirs:
        results_dir = Path(d)
        print(f"\n{'='*70}")
        print(f"  {results_dir}")
        print(f"{'='*70}")

        files = find_result_files(results_dir, args.variants)
        if not files:
            print("  No result files found!")
            continue

        print(f"  {'Variant':<35} {'N':>5}  {'Truth%':>18}  {'Info%':>18}  {'T&I% (conj)':>18}  {'TxI% (prod)':>18}")
        print(f"  {'-'*35} {'-'*5}  {'-'*18}  {'-'*18}  {'-'*18}  {'-'*18}")

        for label, jf in sorted(files.items()):
            truth, info = load_judgments(jf)
            stats = bootstrap_ci(truth, info, args.n_boot, args.ci, args.seed)
            n = len(truth)

            def fmt(s):
                return f"{s['mean']:5.1f} [{s['ci_lo']:5.1f}, {s['ci_hi']:5.1f}]"

            print(f"  {label:<35} {n:>5}  {fmt(stats['truth'])}  {fmt(stats['info'])}  {fmt(stats['t_and_i'])}  {fmt(stats['t_times_i'])}")
            all_results[f"{results_dir.name}/{label}"] = stats

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved to {args.json_out}")


if __name__ == "__main__":
    main()
