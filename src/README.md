# CAA MLP Steering - Source Code

## Pipeline Stages

The pipeline runs as 4 independent stages, each with its own resource requirements:

```
EXTRACT ──> TRAIN ──> GENERATE ──> SCORE
 1 GPU      2 GPU      1 GPU       1 GPU
 ~2hr       ~6hr       ~4hr        ~3hr
```

### Stage 1: Extract (`stages/extract.py`)
Extract steering vectors from model activations (inference only).

```
TruthfulQA ──> Model ──> Activations ──> CAA Vector
```

### Stage 2: Train (`stages/train.py`)
Train MLPs to transform steering vectors (needs gradients).

```
CAA Vector ──> MC-MLP  ──> Optimized for multiple-choice
           ──> Gen-MLP ──> Optimized for generation
```

### Stage 3: Generate (`stages/generate.py`)
Generate responses with steering applied (inference only).

```
Question + Steering ──> Model ──> Response
```

### Stage 4: Score (`stages/score.py`)
Evaluate responses with LLM judges (loads 12B judge model).

```
Responses ──> Truth Judge ──> Scores
          ──> Info Judge
          ──> Semantic Similarity
```

## Architecture Details

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           EXTRACTION STAGE                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   TruthfulQA Dataset                                                     │
│         │                                                                │
│         ▼                                                                │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐               │
│   │ Q + Correct │     │ Q + Wrong   │     │             │               │
│   │   Answer    │     │   Answer    │     │    Model    │               │
│   └──────┬──────┘     └──────┬──────┘     │  (frozen)   │               │
│          │                   │            └──────┬──────┘               │
│          └───────────────────┴───────────────────┘                      │
│                              │                                           │
│                              ▼                                           │
│                    Extract activations at layer L                        │
│                              │                                           │
│                              ▼                                           │
│              ┌───────────────────────────────────┐                      │
│              │  steering_vector = normalize(     │                      │
│              │    mean(correct_acts) -           │                      │
│              │    mean(incorrect_acts)           │                      │
│              │  )                                │                      │
│              └───────────────┬───────────────────┘                      │
│                              │                                           │
│                              ▼                                           │
│                     Base Steering Vector                                 │
│                  (direction of "truthfulness")                           │
└─────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           TRAINING STAGE                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Base Vector ──┬──────────────────────┬─────────────────────────────   │
│                 │                      │                                 │
│                 ▼                      ▼                                 │
│          ┌───────────┐          ┌───────────┐                           │
│          │  MC-MLP   │          │  Gen-MLP  │                           │
│          │  (train)  │          │  (train)  │                           │
│          └─────┬─────┘          └─────┬─────┘                           │
│                │                      │                                  │
│                ▼                      ▼                                  │
│     ┌─────────────────────┐  ┌─────────────────────┐                    │
│     │ Margin/Hinge Loss:  │  │ NLL Loss:           │                    │
│     │ ReLU(logP(wrong) -  │  │ -mean(logP(best_    │                    │
│     │   logP(correct)     │  │   answer tokens))   │                    │
│     │   + margin)         │  │                     │                    │
│     └─────────────────────┘  └─────────────────────┘                    │
│                                                                          │
│   Both use SUPERVISED LEARNING with ground truth from TruthfulQA.        │
└─────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          INFERENCE STAGE                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   For each variant (baseline, steered, mlp_mc, mlp_gen):                │
│                                                                          │
│        Input Question                                                    │
│              │                                                           │
│              ▼                                                           │
│   ┌─────────────────────────────────────────────────────┐               │
│   │                    Model Forward                     │               │
│   │  ┌─────────────────────────────────────────────┐    │               │
│   │  │              Layer L Hook:                   │    │               │
│   │  │  hidden_state += scale * steering_vector    │    │               │
│   │  └─────────────────────────────────────────────┘    │               │
│   └───────────────────────────┬─────────────────────────┘               │
│                               │                                          │
│                               ▼                                          │
│                       Generated Response                                 │
└─────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         EVALUATION STAGE                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Generated Responses                                                    │
│          │                                                               │
│          ├──────────────────┬────────────────────┐                      │
│          ▼                  ▼                    ▼                       │
│   ┌─────────────┐    ┌─────────────┐     ┌─────────────┐                │
│   │ LLM Judge   │    │ LLM Judge   │     │  Semantic   │                │
│   │ (Truthful?) │    │ (Inform.?)  │     │ Similarity  │                │
│   └──────┬──────┘    └──────┬──────┘     └──────┬──────┘                │
│          │                  │                   │                        │
│          └──────────────────┴───────────────────┘                       │
│                             │                                            │
│                             ▼                                            │
│                      Final Metrics                                       │
│                                                                          │
│   LLM judges are ONLY used here during evaluation, not training!         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Module Structure

```
src/
├── stages/           # Independent pipeline stages
│   ├── common.py     # Shared utilities (config, run management)
│   ├── extract.py    # Stage 1: Vector extraction
│   ├── train.py      # Stage 2: MLP training
│   ├── generate.py   # Stage 3: Response generation
│   └── score.py      # Stage 4: Judge evaluation
│
├── steering/         # Core steering logic
│   ├── extract.py    # Activation extraction & CAA computation
│   ├── apply.py      # Forward hook for steering
│   ├── mlp.py        # MLP architecture
│   ├── training.py   # MLP training loops
│   └── vector_bank.py # Bootstrap vector sampling
│
├── evaluation/       # Evaluation judges
│   ├── judge.py      # LLM truthfulness judge
│   ├── informativeness.py  # LLM informativeness judge
│   ├── semantic.py   # Embedding similarity
│   └── truthfulqa.py # MC and generation evaluation
│
├── models/           # Model loading
│   └── loader.py     # HuggingFace model loader
│
├── data/             # Dataset handling
│   └── truthfulqa.py # TruthfulQA dataset manager
│
└── utils/            # Utilities
    ├── config.py     # YAML config loading
    ├── batching.py   # Prompt batching
    └── scoring.py    # Log probability computation
```

## Debugging

Each stage saves diagnostic info:

- **extract**: `raw_activations.pt` with pos/neg activations, `diff_norm_pre_normalize` in metadata
- **train**: `training_history.json` with loss curves, NaN detection
- **generate**: Per-variant response JSONs
- **score**: Individual scored responses + summary

Check stage completion:
```python
from src.stages.common import check_stage_complete, get_or_create_run
ctx = get_or_create_run("gemma3_4b_it", config, run_id)
check_stage_complete(ctx, "extract")  # True/False
```

Re-run a stage:
```bash
python -m src.stages.train --model gemma3_4b_it --run-id <id> --force
```

## Known Issues

### Zero-Norm Vectors
Some instruction-tuned models produce nearly identical activations for pos/neg prompts.

**Symptoms**: `base_vector_norm: 0.0` in metadata, NaN training losses

**Debug**: Check `metadata/extract.json` for `diff_norm_pre_normalize` - if < 1e-6, pos/neg activations are identical.

**Solutions**:
1. Try different layers (earlier layers often differentiate better)
2. Use pretrained model instead of instruction-tuned
3. Increase prompt contrast
