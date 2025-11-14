#!/usr/bin/env python3
"""
Fix judge responses that were truncated due to max_new_tokens=32.
Extract the 'match' field from incomplete JSON responses.
"""

import json
import re
from pathlib import Path
from dataclasses import asdict


def extract_match_from_truncated_json(judge_response: str) -> int:
    """
    Extract match value from truncated JSON response.

    Handles cases like:
    - '```json\n{\n  "match": 1,\n  "explanation": "...'
    - '{"match": 0, "explanation": "...'
    """
    if not judge_response:
        return 0

    # Try to find "match": <value> pattern
    match_pattern = r'"match"\s*:\s*(\d+)'
    match = re.search(match_pattern, judge_response)

    if match:
        return int(match.group(1))

    # Fallback: return 0 if we can't find it
    return 0


def fix_results_file(run_dir: Path, dry_run: bool = False):
    """Fix all generation_details.json files in a run directory."""

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Processing {run_dir.name}")
    print("=" * 80)

    # Find all generation_details.json files
    stats_summary = {}

    for variant_dir in run_dir.iterdir():
        if not variant_dir.is_dir() or variant_dir.name in {"metadata", "vectors"}:
            continue

        variant_name = variant_dir.name

        for scale_dir in variant_dir.iterdir():
            if not scale_dir.is_dir():
                continue

            scale_name = scale_dir.name
            gen_file = scale_dir / "generation_details.json"

            if not gen_file.exists():
                continue

            # Load and fix
            with gen_file.open() as f:
                details = json.load(f)

            fixed_count = 0
            changed_count = 0
            corrected_matches = []

            for item in details:
                judge_response = item.get('judge_response', '')
                old_match = item.get('match', 0)

                # Extract match from response
                new_match = extract_match_from_truncated_json(judge_response)
                corrected_matches.append(new_match)

                if new_match != old_match:
                    changed_count += 1
                    if not dry_run:
                        item['match'] = new_match

                if judge_response and not judge_response.endswith('}'):
                    fixed_count += 1

            # Recompute accuracy
            if details:
                corrected_accuracy = sum(corrected_matches) / len(details)
                semantic_accuracy = sum(d.get('semantic_match', 0) for d in details) / len(details)
            else:
                corrected_accuracy = 0
                semantic_accuracy = 0

            key = f"{variant_name}/{scale_name}"
            stats_summary[key] = {
                'total': len(details),
                'truncated': fixed_count,
                'changed': changed_count,
                'old_accuracy': sum(d.get('match', 0) for d in details) / len(details) if details and dry_run else 0,
                'corrected_accuracy': corrected_accuracy,
                'semantic_accuracy': semantic_accuracy,
            }

            # Save if not dry run
            if not dry_run:
                with gen_file.open('w') as f:
                    json.dump(details, f, indent=2)

    # Print summary
    for key, stats in sorted(stats_summary.items()):
        print(f"\n{key}:")
        print(f"  Total responses: {stats['total']}")
        print(f"  Truncated responses: {stats['truncated']}")
        print(f"  Match values changed: {stats['changed']}")
        if dry_run:
            print(f"  Old accuracy (broken): {stats['old_accuracy']:.2%}")
        print(f"  Corrected accuracy: {stats['corrected_accuracy']:.2%}")
        print(f"  Semantic accuracy: {stats['semantic_accuracy']:.2%}")

    # Update results.json with corrected stats
    if not dry_run:
        results_file = run_dir / "results.json"
        if results_file.exists():
            with results_file.open() as f:
                results = json.load(f)

            # Update accuracy in stats
            for variant_name in results:
                if variant_name in {"metadata", "vectors"}:
                    continue
                for scale_name in results[variant_name]:
                    if 'generation' in results[variant_name][scale_name]:
                        key = f"{variant_name}/{scale_name}"
                        if key in stats_summary:
                            results[variant_name][scale_name]['generation']['stats']['accuracy'] = \
                                stats_summary[key]['corrected_accuracy']

            with results_file.open('w') as f:
                json.dump(results, f, indent=2)

            print(f"\n✓ Updated {results_file}")


if __name__ == "__main__":
    import sys

    base_dir = Path("analysis_results")

    # Check if specific directory provided as argument
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        run_dir = Path(sys.argv[1])
        runs = [run_dir] if run_dir.exists() else []
        if not runs:
            print(f"❌ Directory not found: {sys.argv[1]}")
            sys.exit(1)
    else:
        # Process all known models with truncated JSON issues
        runs = [
            base_dir / "gemma2_27b_full_20251106_091841",
            base_dir / "gemma3_27b_full_20251106_101727",
            base_dir / "gemma3_4b_full_20251107_172642",
        ]

    dry_run = "--dry-run" in sys.argv

    if dry_run:
        print("\n" + "=" * 80)
        print("DRY RUN MODE - No files will be modified")
        print("=" * 80)

    for run_dir in runs:
        if run_dir.exists():
            fix_results_file(run_dir, dry_run=dry_run)
        else:
            print(f"\nSkipping {run_dir.name} (not found)")

    if dry_run:
        print("\n" + "=" * 80)
        print("To apply changes, run without --dry-run flag")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("✓ All results fixed!")
        print("=" * 80)
