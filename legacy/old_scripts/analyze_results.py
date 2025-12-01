#!/usr/bin/env python3
"""Analyze TruthfulQA CAA-MLP steering results."""

import json
import sys
from pathlib import Path
from collections import defaultdict

def load_results(run_dir: Path) -> dict:
    results_file = run_dir / "results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    with open(results_file) as f:
        return json.load(f)

def load_generation_details(variant: str, scale: float, run_dir: Path) -> list:
    details_file = run_dir / variant / f"scale_{scale:.2f}" / "generation_details.json"
    if not details_file.exists():
        return []

    with open(details_file) as f:
        return json.load(f)

def analyze_verbosity(details: list) -> dict:
    """Analyze answer length and verbosity patterns."""
    if not details:
        return {}

    lengths = [len(d["generated"].split()) for d in details]
    has_followup = sum(1 for d in details if "\nQuestion:" in d["generated"] or "\n\n" in d["generated"])

    return {
        "avg_words": sum(lengths) / len(lengths),
        "min_words": min(lengths),
        "max_words": max(lengths),
        "pct_with_followup": 100 * has_followup / len(details),
    }

def compare_examples(baseline_details, mlp_gen_details, n=5):
    """Find interesting comparison examples."""
    improvements = []
    degradations = []

    for b, m in zip(baseline_details, mlp_gen_details):
        if b["question"] != m["question"]:
            continue

        baseline_score = b.get("semantic_score", 0)
        mlp_score = m.get("semantic_score", 0)
        delta = mlp_score - baseline_score

        example = {
            "question": b["question"],
            "baseline": b["generated"][:100],
            "mlp_gen": m["generated"][:100],
            "baseline_match": b.get("match", 0),
            "mlp_match": m.get("match", 0),
            "semantic_delta": delta,
        }

        if delta > 0.1:
            improvements.append(example)
        elif delta < -0.1:
            degradations.append(example)

    improvements.sort(key=lambda x: x["semantic_delta"], reverse=True)
    degradations.sort(key=lambda x: x["semantic_delta"])

    return improvements[:n], degradations[:n]

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <run_directory>")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    results = load_results(run_dir)

    print("=" * 80)
    print("TRUTHFULQA CAA-MLP STEERING ANALYSIS")
    print("=" * 80)
    print(f"\nRun: {run_dir.name}\n")

    # Main results table
    print("## ACCURACY & SEMANTIC SCORES\n")
    print(f"{'Variant':<20} {'Judge Acc':<12} {'Semantic':<12} {'Δ Acc':<12} {'Δ Sem':<12}")
    print("-" * 80)

    baseline_gen = results["baseline"]["scale_0.00"]["generation"]
    baseline_acc = baseline_gen["accuracy"]
    baseline_sem = baseline_gen["semantic_mean"]

    for variant in ["baseline", "steered", "mlp_mc", "mlp_gen"]:
        scale_key = "scale_0.00" if variant == "baseline" else "scale_1.00"
        gen = results[variant][scale_key]["generation"]

        acc = gen["accuracy"]
        sem = gen["semantic_mean"]
        delta_acc = ((acc - baseline_acc) / baseline_acc * 100) if baseline_acc > 0 else 0
        delta_sem = ((sem - baseline_sem) / baseline_sem * 100) if baseline_sem > 0 else 0

        print(f"{variant:<20} {acc:<12.3f} {sem:<12.3f} {delta_acc:+11.1f}% {delta_sem:+11.1f}%")

    # Verbosity analysis
    print("\n## VERBOSITY ANALYSIS\n")
    print(f"{'Variant':<20} {'Avg Words':<12} {'Min':<8} {'Max':<8} {'% Follow-up':<12}")
    print("-" * 80)

    for variant in ["baseline", "mlp_gen"]:
        scale = 0.0 if variant == "baseline" else 1.0
        details = load_generation_details(variant, scale, run_dir)
        if not details:
            continue

        verb = analyze_verbosity(details)
        print(f"{variant:<20} {verb['avg_words']:<12.1f} {verb['min_words']:<8} {verb['max_words']:<8} {verb['pct_with_followup']:<11.1f}%")

    # Example comparisons
    print("\n## NOTABLE EXAMPLES\n")

    baseline_details = load_generation_details("baseline", 0.0, run_dir)
    mlp_gen_details = load_generation_details("mlp_gen", 1.0, run_dir)

    if baseline_details and mlp_gen_details:
        improvements, degradations = compare_examples(baseline_details, mlp_gen_details, n=3)

        if improvements:
            print("### Top 3 Improvements (MLP-Gen > Baseline):\n")
            for i, ex in enumerate(improvements, 1):
                print(f"{i}. **Q:** {ex['question'][:70]}...")
                print(f"   Semantic Δ: +{ex['semantic_delta']:.3f}")
                print(f"   Judge: {ex['baseline_match']} → {ex['mlp_match']}")
                print(f"   Baseline: {ex['baseline']}...")
                print(f"   MLP-Gen:  {ex['mlp_gen']}...\n")

        if degradations:
            print("### Top 3 Degradations (MLP-Gen < Baseline):\n")
            for i, ex in enumerate(degradations, 1):
                print(f"{i}. **Q:** {ex['question'][:70]}...")
                print(f"   Semantic Δ: {ex['semantic_delta']:.3f}")
                print(f"   Judge: {ex['baseline_match']} → {ex['mlp_match']}")
                print(f"   Baseline: {ex['baseline']}...")
                print(f"   MLP-Gen:  {ex['mlp_gen']}...\n")

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    mlp_gen = results["mlp_gen"]["scale_1.00"]["generation"]
    mlp_acc = mlp_gen["accuracy"]
    mlp_sem = mlp_gen["semantic_mean"]

    print(f"\n✅ MLP-Gen Judge Accuracy:     {mlp_acc:.1%} ({mlp_acc - baseline_acc:+.1%} vs baseline)")
    print(f"✅ MLP-Gen Semantic Score:     {mlp_sem:.3f} ({mlp_sem - baseline_sem:+.3f} vs baseline)")
    print(f"\n📊 Total examples evaluated:   {baseline_gen['total']}")
    print(f"🎯 Best performing variant:    {'MLP-Gen' if mlp_acc >= baseline_acc else 'Baseline'} (judge accuracy)")

    if mlp_sem > baseline_sem and mlp_acc < baseline_acc:
        print(f"\n⚠️  DISCREPANCY: Semantic score improved (+{((mlp_sem/baseline_sem - 1) * 100):.1f}%) but judge accuracy declined ({((mlp_acc/baseline_acc - 1) * 100):.1f}%)")
        print("    This suggests more verbose/detailed answers that miss key facts.")
    elif mlp_acc > baseline_acc:
        print(f"\n🎉 SUCCESS: Both metrics improved!")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
