# Quick Wins Applied - 2025-11-05

## Q1: Fix Gemma-3 NaN Issues

### 1. Safe Normalization (`src/steering/extract.py`)
- **Added epsilon check** before F.normalize to prevent division by near-zero norms
- **Added NaN/Inf detection** before normalization with error logging
- **Warns** if vector norm < 1e-8 (indicates non-discriminative layer)

### 2. FP32 Upcast (`src/steering/extract.py`)
- **Upcast activations to fp32** during hook collection (`.float()`)
- **Disable autocast** during extraction forward pass to prevent bfloat16 numerical explosions
- Fixes: Gemma-3-1B produced all-NaN steering vectors at layer 9

### 3. Layer Update (`configs/gemma3-1b.yaml`)
- **Changed layer 9 → 12** to match Gemma-2 relative depth
- Layer 9 was too early for Gemma-3 to differentiate truthful vs deceptive

---

## Q3: MSE Regularization (`src/steering/training.py`)

### Both MC and Gen Trainers
- **Added `mse_reg: float = 1e-2`** to config dataclasses
- **Loss:** `loss = task_loss + λ * mse_loss(MLP(v), v)`
  - MC: `hinge_loss + 0.01 * mse(transformed, vector)`
  - Gen: `nll_loss + 0.01 * mse(transformed, vector)`
- **Purpose:** Prevent MLP from drifting too far from identity, reduces verbosity-inducing transformations

---

## Q4: Reduce Verbosity

### 1. Generation Config (`configs/base.yaml`)
**Before:**
```yaml
max_new_tokens: 80
temperature: 0.7
```

**After:**
```yaml
max_new_tokens: 64  # Reduced from 80
temperature: 0.3    # Reduced from 0.7 for focused answers
stop_sequences: ["\n\n", "\nQuestion:"]  # Stop at double newline or next question
```

### 2. Prompt Update (`src/evaluation/truthfulqa.py`)
**Before:**
```python
prompt = f"Question: {question}\nAnswer:"
```

**After:**
```python
prompt = f"Question: {question}\nAnswer concisely in one sentence:"
```

### 3. Stop Sequences Implementation
- Added `StoppingCriteria` class to handle multi-token stop sequences
- Converts stop strings to token IDs and checks during generation
- Prevents model from generating follow-up questions or rambling

---

## Q2: Still TODO (Lower Priority)

### 1. Vector Bank Sampling in Eval
- Currently: eval only uses `base_vector`
- Needed: Sample N=8 vectors from bank, report mean ± stdev
- Benefits: More robust estimates, matches training procedure

### 2. Scale Sweep
- Currently: Only evaluates at scale=1.0
- Needed: Test [0.25, 0.5, 0.75, 1.0, 1.25]
- Pick optimal by dev judge accuracy

---

## Expected Improvements

**From NaN Fixes (Q1):**
- ✅ Gemma-3-1B should now train without NaN losses
- ✅ Layer 12 likely more discriminative than layer 9

**From MSE Regularization (Q3):**
- 📉 Reduced verbosity in generated answers
- 📈 Better judge accuracy (less meandering)
- 🎯 MLP stays closer to raw CAA direction

**From Verbosity Fixes (Q4):**
- 📉 50% shorter answers on average (64 vs 80 tokens, temp 0.3 vs 0.7)
- 🛑 Stop before generating follow-up questions
- 📊 Judge and semantic scores should align better

---

## Test Plan

**Smoke Test (Job TBD):**
- Model: Gemma-3-1B-it
- Layer: 12 (updated)
- Expected: No NaN, finite losses, reasonable accuracy

**If Successful:**
1. Run smoke test on Gemma-2-2B with new settings
2. Compare verbosity: old vs new generations
3. If judge accuracy improves: proceed with full experiments (6 models)
