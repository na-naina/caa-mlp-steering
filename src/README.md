# Pipeline Architecture

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
│   No RL or reward models involved in training.                           │
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

## Key Components

**Steering Vector Extraction** (`steering/extract.py`)

- Run model on truthful vs untruthful completions
- Extract hidden states at target layer
- Compute normalized difference as "truthfulness direction"

**MLP Training** (`steering/training.py`) - Supervised Learning, NOT RL!

- **MC-MLP**: Optimizes for multiple-choice by maximizing logprob gap between correct and incorrect answers using margin/hinge loss
- **Gen-MLP**: Optimizes for generation by minimizing NLL on ground-truth best answers
- Both MLPs preserve vector norm to avoid destabilizing residual stream

**Steering Application** (`steering/apply.py`)

- Register forward hook at target layer
- Add scaled steering vector to hidden states during generation

**Evaluation** (`evaluation/`)

- Zero-shot LLM judges assess truthfulness and informativeness
- Semantic similarity compares to reference answers
