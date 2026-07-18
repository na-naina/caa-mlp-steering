#!/usr/bin/env python3
"""Judge the CAA α-sweep generations using existing GPT-4o-mini truth/info judges."""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Re-use the helpers from the main judge script.
from scripts.evaluate_with_gpt_judge import (
    clean_response,
    evaluate_single,
    SYSTEM_PROMPT_TRUTH,
    SYSTEM_PROMPT_INFO,
)
from dotenv import load_dotenv

load_dotenv()

TRUTHFULQA_DIR = ROOT / "data" / "truthfulqa"


def get_judge_model_ids():
    """Fetch the fine-tuned judge model IDs from OpenAI for the recorded job IDs."""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    models = {}
    for jt in ("truth", "info"):
        info = json.loads((TRUTHFULQA_DIR / f"finetune_job_{jt}_gpt-4o-mini.json").read_text())
        job = client.fine_tuning.jobs.retrieve(info["job_id"])
        if job.status != "succeeded" or not job.fine_tuned_model:
            raise RuntimeError(f"{jt} judge job not ready: status={job.status}")
        models[jt] = job.fine_tuned_model
        print(f"  {jt}: {job.fine_tuned_model}", flush=True)
    return models


def judge_one(args):
    idx, ex, judges, client = args
    generated, was_truncated = clean_response(ex["generated"])
    truth = evaluate_single(client, judges["truth"], ex["question"], generated, SYSTEM_PROMPT_TRUTH)
    info = evaluate_single(client, judges["info"], ex["question"], generated, SYSTEM_PROMPT_INFO)
    return idx, {
        "question": ex["question"],
        "generated_original": ex["generated"],
        "generated_clean": generated,
        "was_truncated": was_truncated,
        "best_answer": ex.get("best_answer", ""),
        "truth_judgment": truth,
        "info_judgment": info,
    }, was_truncated


def judge_file(gen_path: Path, judges: dict, out_path: Path, n_workers: int = 8):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    items = json.loads(gen_path.read_text())
    n = len(items)
    results = [None] * n
    args_list = [(i, ex, judges, client) for i, ex in enumerate(items)]
    n_trunc = 0
    completed = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(judge_one, a): a[0] for a in args_list}
        for fut in futures:
            pass
        from concurrent.futures import as_completed
        for fut in as_completed(futures):
            i, res, trunc = fut.result()
            results[i] = res
            if trunc:
                n_trunc += 1
            completed += 1
            if completed % 50 == 0:
                el = time.time() - t0
                rate = el / completed
                print(f"    [{gen_path.name}] {completed}/{n} ({rate:.2f}s/ex)", flush=True)

    truth_yes = sum(1 for r in results if r["truth_judgment"] == "yes")
    info_yes = sum(1 for r in results if r["info_judgment"] == "yes")
    ti_yes = sum(1 for r in results if r["truth_judgment"] == "yes" and r["info_judgment"] == "yes")

    summary = {
        "stats": {
            "truth_accuracy": truth_yes / n,
            "info_accuracy": info_yes / n,
            "truth_and_info_accuracy": ti_yes / n,
            "truth_yes": truth_yes,
            "info_yes": info_yes,
            "truth_and_info": ti_yes,
            "total": n,
            "n_truncated": n_trunc,
            "truncation_rate": n_trunc / n,
            "judge_models": judges,
            "clean_mode": True,
        },
        "results": results,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    s = summary["stats"]
    print(f"  → Truth={s['truth_accuracy']*100:.2f}  Info={s['info_accuracy']*100:.2f}  T×I={s['truth_and_info_accuracy']*100:.2f}", flush=True)
    return summary["stats"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", type=Path, default=ROOT / "data" / "outputs" / "caa_alpha_sweep_seed42")
    p.add_argument("--out_dir", type=Path, default=None)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    out_dir = args.out_dir or args.in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching judge model IDs…", flush=True)
    judges = get_judge_model_ids()
    print()

    all_stats = {}
    for gen in sorted(args.in_dir.glob("alpha_[0-9]*.json")):
        if gen.name.endswith("_judged.json") or gen.name == "alpha_sweep_summary.json":
            continue
        out = out_dir / f"{gen.stem}_judged.json"
        if out.exists():
            print(f"== {gen.name} already judged at {out.name}, skipping ==", flush=True)
            all_stats[gen.name] = json.loads(out.read_text())["stats"]
            continue
        print(f"== Judging {gen.name} ==", flush=True)
        all_stats[gen.name] = judge_file(gen, judges, out, n_workers=args.workers)

    print("\nSummary:")
    print(f"{'alpha':>8s}  {'Truth%':>7s} {'Info%':>7s} {'T×I%':>7s} {'n_trunc':>7s}")
    for name, s in sorted(all_stats.items()):
        print(f"{name:>14s}  {s['truth_accuracy']*100:7.2f} {s['info_accuracy']*100:7.2f} {s['truth_and_info_accuracy']*100:7.2f} {s['n_truncated']:>7d}")

    (out_dir / "alpha_sweep_summary.json").write_text(json.dumps(all_stats, indent=2))
    print(f"\nWrote {out_dir / 'alpha_sweep_summary.json'}")


if __name__ == "__main__":
    main()
