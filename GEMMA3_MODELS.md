# Gemma-3 Model Variants

## Available Sizes

| Size | Parameters | Type | Context | Model ID (IT) |
|------|-----------|------|---------|---------------|
| **270M** | 270M | Text-only | 32K | `google/gemma-3-270m-it` |
| **1B** | 1B | Text-only | 32K | `google/gemma-3-1b-it` |
| **4B** | 4B | Multimodal | 128K | `google/gemma-3-4b-it` |
| **12B** | 12B | Multimodal | 128K | `google/gemma-3-12b-it` |
| **27B** | 27B | Multimodal | 128K | `google/gemma-3-27b-it` |

## Important Notes

1. **No 9B variant exists** - Previous configs referencing `gemma-3-9b` are incorrect

2. **Multimodal = Text-compatible**: The 4B, 12B, and 27B models can process images+text BUT work perfectly for text-only tasks without loading vision encoder

3. **Context improvements**:
   - 1B: 32K tokens (4x vs Gemma-2's 8K)
   - 4B/12B/27B: 128K tokens (16x vs Gemma-2)

4. **Multilingual**: 140+ languages supported

## Recommended for Our Experiments

For TruthfulQA CAA experiments (text-only):

- **Gemma-3-1B-it**: Smallest, fastest, text-only
- **Gemma-3-12B-it**: ✅ Already used as judge, good mid-size option
- **Gemma-3-27B-it**: Largest, best performance (requires 2 GPUs)

**Skip 4B** - it's between 1B and 12B without clear advantage for our use case.

## Config Updates Needed

Current configs reference non-existent models:
- ❌ `configs/gemma3-9b.yaml` → Should use `google/gemma-3-12b-it` instead
- ✅ `configs/gemma3-1b.yaml` → Correct
- ✅ `configs/gemma3-27b.yaml` → Correct

## Usage Example

```python
from transformers import AutoModelForCausalLM

# Text-only usage of multimodal model
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-12b-it",
    device_map="auto",
    torch_dtype="bfloat16"
)
# Vision encoder NOT loaded - memory efficient!
```
