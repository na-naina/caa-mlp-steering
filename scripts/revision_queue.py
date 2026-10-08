#!/usr/bin/env python3
"""Emit the October 2026 revision run queue as two job files (NAME<TAB>COMMAND).

Protocol: RaLFiT-aligned 2-fold cross-validation over all 817 TruthfulQA
questions (scripts/make_cv2_splits.py), repeated for 3 seeds; every method sees
the same splits, decoding (T=0.3) and judges. Output layout is
data/outputs/rcv_<method>_s<seed>/fold<k>/<variant>/scale_x/, which
scripts/evaluate_with_gpt_judge.py discovers automatically.

Training and generation are separate jobs so they can use different GPU
layouts: on 32 GB cards, 7B training (batch 8) must be sharded over two GPUs,
while generation fits two jobs per card.

Usage:
    python scripts/revision_queue.py --out-prefix jobs
    bash scripts/jobqueue.sh jobs_train.txt 0    # one job at a time, all GPUs
    bash scripts/jobqueue.sh jobs_gen.txt 2      # two jobs per GPU
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

PY = ".venv/bin/python"
LLAMA = "llama2_7b_chat_L8_bn8"
WITH_CAA = "--set 'steering.enabled_variants=[baseline, steered, mlp_mc]'"
MAST_ONLY = "--set 'steering.enabled_variants=[mlp_mc]'"


@dataclass
class Exp:
    name: str
    prio: int          # 0 core table, 1 lr curves, 2 cross-model / thesis ablations
    kind: str          # run | dv | lora
    out: str
    args: str          # method arguments shared by the train and generate commands
    split: str = ""    # --splits-file (training only; generation reuses metadata/splits.json)
    seed: int = 42
    torch_seed: int | None = None

    def seeds(self) -> str:
        ts = f" --torch-seed {self.torch_seed}" if self.torch_seed is not None else ""
        return f"--seed {self.seed}{ts}"

    def train_cmd(self) -> str:
        spl = f" --splits-file {self.split}" if self.split else ""
        if self.kind == "run":
            return (f"{PY} run.py --stage train-only {self.seeds()}{spl} "
                    f"--output-dir {self.out} {self.args}")
        if self.kind == "dv":
            return (f"{PY} scripts/train_direct_vector.py --skip-generation {self.seeds()}{spl} "
                    f"--output-dir {self.out} {self.args}")
        return (f"{PY} scripts/train_lora_dpo.py --train-only {self.seeds()}{spl} "
                f"--output-dir {self.out} {self.args}")

    def gen_cmd(self) -> str:
        if self.kind == "run":
            return f"{PY} run.py --stage generate {self.seeds()} --run-dir {self.out} {self.args}"
        if self.kind == "dv":
            return (f"{PY} scripts/train_direct_vector.py --generate-only {self.seeds()} "
                    f"--output-dir {self.out} {self.args}")
        return (f"{PY} scripts/train_lora_dpo.py --skip-lora --lora-only {self.seeds()} "
                f"--splits-file {self.split} --output-dir {self.out} {self.args}")


def cell(seed: int, fold: int):
    spl = f"data/splits/cv2_s{seed}_fold{fold}.json"
    out = lambda m: f"data/outputs/rcv_{m}_s{seed}/fold{fold}"  # noqa: E731
    tag = f"s{seed}f{fold}"
    kw = dict(split=spl, seed=seed)

    # Baseline, raw CAA (alpha 1 and the tuned alpha 2), MAST at its default lr
    yield Exp(f"main_{tag}", 0, "run", out("main"),
              f"--model {LLAMA} {WITH_CAA} --set 'steering.caa_scales=[1.0, 2.0]'", **kw)
    # Directly optimised vector: zero init at its oracle lr, CAA init at MAST's lr
    yield Exp(f"dvzero_lr2e-3_{tag}", 0, "dv", out("dvzero_lr2e-3"), "--init zero --lr 2e-3", **kw)
    yield Exp(f"dvcaa_lr5e-4_{tag}", 0, "dv", out("dvcaa_lr5e-4"), "--init caa --lr 5e-4", **kw)
    # Weight-space reference on the identical training half (RaLFiT's LoRA-DPO setting:
    # W_O + W_down, rank 8, DPO, LoFiT batch 8 / 5 epochs)
    yield Exp(f"loradpo_{tag}", 0, "lora", out("loradpo"),
              "--include-pool --target-modules o_proj,down_proj --lora-r 8 --lr 1e-4 "
              "--epochs 5 --batch-size 8", **kw)
    # Learning-rate curves: MAST vs direct vector at every lr
    # (YAML 1.1 reads "1e-3" as a string, so pass the lr in decimal form)
    for lr, dec in (("1e-3", "0.001"), ("2e-3", "0.002")):
        yield Exp(f"mast_lr{lr}_{tag}", 1, "run", out(f"mast_lr{lr}"),
                  f"--model {LLAMA} {MAST_ONLY} --set mlp.mc_training.lr={dec}", **kw)
    for lr in ("5e-4", "1e-3", "5e-3"):
        yield Exp(f"dvzero_lr{lr}_{tag}", 1, "dv", out(f"dvzero_lr{lr}"), f"--init zero --lr {lr}", **kw)
    yield Exp(f"dvcaa_lr2e-3_{tag}", 1, "dv", out("dvcaa_lr2e-3"), "--init caa --lr 2e-3", **kw)


def extras():
    # Cross-model tie-breaker: does the bare vector need per-model lr tuning where MAST does not?
    g = "--model google/gemma-3-4b-it --layer 13"
    for init, lr in (("zero", "5e-4"), ("zero", "2e-3"), ("zero", "5e-3"), ("caa", "5e-4")):
        yield Exp(f"g4b_dv{init}_lr{lr}", 2, "dv", f"data/outputs/rg4b_dv{init}_lr{lr}_s42/fold1",
                  f"{g} --init {init} --lr {lr}", split="data/splits/cv2_s42_fold1.json")
    # Second CV fold for the Gemma transfer cell (fold1 = existing g4b_bn8_full, seed 42)
    yield Exp("g4b_main_s42f2", 2, "run", "data/outputs/rcv_g4bmain_s42/fold2",
              "--model gemma3_4b_bn8_L13", split="data/splits/cv2_s42_fold2.json")
    # Examiner correction 5: multi-seed noise-input ablation (fold1 of each seed)
    for seed in (42, 123, 456):
        yield Exp(f"noise_s{seed}", 2, "run", f"data/outputs/rcv_noise_s{seed}/fold1",
                  f"--model {LLAMA} {MAST_ONLY} --set steering.ablation=noise",
                  split=f"data/splits/cv2_s{seed}_fold1.json", seed=seed)
    # Category hold-out repeats (torch seeds; the category split itself is fixed)
    for fold in ("A", "B"):
        for ts in (123, 456):
            yield Exp(f"cathold_{fold}_ts{ts}", 2, "run", f"data/outputs/rcathold_{fold}_ts{ts}",
                      f"--model {LLAMA} {WITH_CAA}",
                      split=f"data/splits/cat_holdout/fold_{fold}/splits.json", torch_seed=ts)


def mc_jobs(seed: int, fold: int):
    """Canonical MC1/MC2 (lm-eval-harness): baseline, raw CAA, MAST, tuned direct vector."""
    out = lambda m: f"data/outputs/rcv_{m}_s{seed}/fold{fold}"  # noqa: E731
    tag = f"s{seed}f{fold}"
    h = f"{PY} scripts/eval_mc_harness.py"
    yield f"mc_main_{tag}", (f"{h} --run-dir {out('main')} --vector-file vectors/v_mlp_mc.pt "
                             f"--variants baseline steered --label mast")
    yield f"mc_caa_{tag}", (f"{h} --run-dir {out('main')} --vector-file vectors/v_steered.pt "
                            f"--variants steered --label raw_caa --output {out('main')}/mc_harness_caa.json")
    yield f"mc_dvzero_{tag}", (f"{h} --run-dir {out('dvzero_lr2e-3')} --vector-file vectors/optimized_vector.pt "
                               f"--variants steered --label dvzero")


# ---------------------------------------------------------------------------
# 8 Oct overnight blocks (single 96 GB card). Each job yields (name, train, generate);
# write_block emits <prefix>_train.txt (7B training peaks ~40 GB -> 2 slots) and
# <prefix>_gen.txt (~13 GB per generation job -> 5 slots).
# ---------------------------------------------------------------------------
PY5 = ".venv-tf5/bin/python"   # transformers 5.x env for Gemma-4 / Qwen3.5
# Paper recipe on one card: batch 8, no accumulation, 2 x 50 steps (the tf5 configs
# default to 4 x accum 2 for 32 GB cards)
FULL_BATCH = ("--set mlp.mc_training.batch_size=8 --set mlp.mc_training.gradient_accumulation_steps=1 "
              "--set mlp.mc_training.steps_per_epoch=50")
NEW_MODELS = {
    # key: (python, config prefix, HF id, LoRA targets)
    "g4e": (PY5, "gemma4_e4b_bn8_L", "google/gemma-4-E4B-it", "'re:.*language_model.*\\.(o_proj|down_proj)'"),
    "q35": (PY5, "qwen3_5_9b_bn8_L", "Qwen/Qwen3.5-9B", "o_proj,out_proj,down_proj"),
}
CELLS = [(s, f) for s in (42, 123, 456) for f in (1, 2)]


def both(e: Exp, py: str = PY):
    return e.train_cmd().replace(PY, py, 1), e.gen_cmd().replace(PY, py, 1)


def block_p0():
    for s, f in CELLS:
        spl, tag = f"data/splits/cv2_s{s}_fold{f}.json", f"s{s}f{f}"
        for lr, dec in (("1e-4", "0.0001"), ("2e-4", "0.0002"), ("3e-4", "0.0003")):
            e = Exp(f"mast_lr{lr}_{tag}", 0, "run", f"data/outputs/rcv_mast_lr{lr}_s{s}/fold{f}",
                    f"--model {LLAMA} {MAST_ONLY} --set mlp.mc_training.lr={dec}", split=spl, seed=s)
            yield (e.name, *both(e))
    for s in (123, 456):
        for f in (1, 2):
            for lr in ("5e-4", "2e-3"):
                yield (f"bipo_lr{lr}_s{s}f{f}", "",
                       f"{PY} scripts/train_direct_vector.py --generate-only --loss bipo --seed {s} "
                       f"--output-dir data/outputs/rcv_bipo_lr{lr}_s{s}/fold{f}")


def block_sweep(key: str, layers: list[int]):
    py, cfg, _, _ = NEW_MODELS[key]
    for L in layers:
        yield (f"{key}_sweep_L{L}",
               f"{py} run.py --stage train-only --seed 42 --splits-file data/splits/cv2_s42_fold1.json "
               f"--output-dir data/outputs/r{key}_sweep/L{L} --model {cfg}{L} {MAST_ONLY} {FULL_BATCH}", "")


def block_new_model(key: str, layer: int, seeds=(42, 123, 456), pred_lr: str | None = None):
    py, cfg, hf, targets = NEW_MODELS[key]
    for s in seeds:
        for f in (1, 2):
            spl, tag = f"data/splits/cv2_s{s}_fold{f}.json", f"s{s}f{f}"
            out = lambda m: f"data/outputs/rcv_{key}{m}_s{s}/fold{f}"  # noqa: E731
            kw = dict(split=spl, seed=s)
            exps = [
                Exp(f"{key}_main_{tag}", 0, "run", out("main"),
                    f"--model {cfg}{layer} {WITH_CAA} --set 'steering.caa_scales=[1.0, 2.0]' {FULL_BATCH}", **kw),
                Exp(f"{key}_dvscaled_{tag}", 0, "dv", out("_dvscaled_lr8e-4"),
                    f"--model {hf} --layer {layer} --init zero --scale-by-caa --lr 8e-4", **kw),
                Exp(f"{key}_loradpo_{tag}", 0, "lora", out("_loradpo"),
                    f"--model {hf} --include-pool --target-modules {targets} --lora-r 8 --lr 1e-4 "
                    f"--epochs 5 --batch-size 8", **kw),
            ]
            if pred_lr and s == 42 and f == 1:
                exps.append(Exp(f"{key}_dvpred_{tag}", 0, "dv", out(f"_dvzero_lr{pred_lr}"),
                                f"--model {hf} --layer {layer} --init zero --lr {pred_lr}", **kw))
            for e in exps:
                yield (e.name, *both(e, py))


def block_p3():
    # Older 2025 models filled to 3 seeds x 2 folds (gemma3 s42 fold1/2 and qwen3 s42 fold1 exist)
    olds = {"g4b": ("gemma3_4b_bn8_L13", "google/gemma-3-4b-it", 13, ""),
            "q4b": ("qwen3_4b_bn8", "Qwen/Qwen3-4B", 14, " --set model.layer=14")}
    for key, (cfg, hf, layer, extra) in olds.items():
        for s, f in CELLS:
            spl, tag = f"data/splits/cv2_s{s}_fold{f}.json", f"s{s}f{f}"
            kw = dict(split=spl, seed=s)
            skip_main = (key == "g4b" and s == 42) or (key == "q4b" and (s, f) == (42, 1))
            if not skip_main:
                e = Exp(f"{key}_main_{tag}", 2, "run", f"data/outputs/rcv_{key}main_s{s}/fold{f}",
                        f"--model {cfg}{extra} {WITH_CAA} --set 'steering.caa_scales=[1.0, 2.0]'", **kw)
                yield (e.name, *both(e))
            e = Exp(f"{key}_dvscaled_{tag}", 2, "dv", f"data/outputs/rcv_{key}_dvscaled_lr8e-4_s{s}/fold{f}",
                    f"--model {hf} --layer {layer} --init zero --scale-by-caa --lr 8e-4", **kw)
            yield (e.name, *both(e))


def write_block(name: str, jobs, path: str):
    jobs = list(jobs)
    stem = path[:-4] if path.endswith(".txt") else path
    for suffix, k in (("_train.txt", 1), ("_gen.txt", 2)):
        with open(stem + suffix, "w") as fh:
            for j in jobs:
                if j[k]:
                    fh.write(f"{j[0]}\t{j[k]}\n")
    print(f"{name}: {len(jobs)} jobs -> {stem}_train.txt / {stem}_gen.txt")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    p.add_argument("--max-priority", type=int, default=2)
    p.add_argument("--out-prefix", default="jobs")
    p.add_argument("--block", choices=["p0", "sweep", "new", "p3"],
                   help="emit one 8-Oct overnight block (combined train+generate jobs)")
    p.add_argument("--key", choices=list(NEW_MODELS), help="new-model key for --block sweep/new")
    p.add_argument("--layers", type=int, nargs="+", help="sweep layers, or the chosen layer for --block new")
    p.add_argument("--pred-lr", help="bare-vector lr predicted by the activation-norm rule (seed 42 fold 1)")
    p.add_argument("--block-seeds", type=int, nargs="+", default=[42, 123, 456])
    args = p.parse_args()
    if args.block:
        jobs = {"p0": lambda: block_p0(),
                "sweep": lambda: block_sweep(args.key, args.layers),
                "new": lambda: block_new_model(args.key, args.layers[0], tuple(args.block_seeds), args.pred_lr),
                "p3": lambda: block_p3()}[args.block]()
        write_block(args.block, jobs, f"{args.out_prefix}.txt")
        return

    cells = [(s, f) for s in args.seeds for f in (1, 2)]
    exps = [e for s, f in cells for e in cell(s, f)] + list(extras())
    exps = sorted((e for e in exps if e.prio <= args.max_priority), key=lambda e: e.prio)

    with open(f"{args.out_prefix}_train.txt", "w") as fh:
        for e in exps:
            fh.write(f"{e.name}\t{e.train_cmd()}\n")
    with open(f"{args.out_prefix}_gen.txt", "w") as fh:
        for e in exps:
            fh.write(f"{e.name}\t{e.gen_cmd()}\n")
        for s, f in cells:
            for name, cmd in mc_jobs(s, f):
                fh.write(f"{name}\t{cmd}\n")
    print(f"{len(exps)} experiments -> {args.out_prefix}_train.txt / {args.out_prefix}_gen.txt")


if __name__ == "__main__":
    main()
