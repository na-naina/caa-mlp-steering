#!/usr/bin/env python3
"""Debug script to verify extraction works for different model architectures."""
import torch
import torch.nn as nn
from contextlib import contextmanager

# Minimal mock models to test architecture detection

class MockTextModel(nn.Module):
    """Simulates Gemma3ForCausalLM (text-only, 1B)"""
    def __init__(self, num_layers=4, hidden_size=64):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_size, 100)

class MockLanguageModel(nn.Module):
    """Inner language model for multimodal"""
    def __init__(self, num_layers=4, hidden_size=64):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])

class MockMultimodalModel(nn.Module):
    """Simulates Gemma3ForConditionalGeneration (multimodal, 4B+)"""
    def __init__(self, num_layers=4, hidden_size=64):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = MockLanguageModel(num_layers, hidden_size)
        # No direct model.model.layers attribute
        self.lm_head = nn.Linear(hidden_size, 100)


def _get_decoder_layers(model) -> list:
    """Copy of the function from extract.py"""
    # Standard text-only architecture (Gemma, Gemma2, Gemma3ForCausalLM, etc.)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # Multimodal architecture (Gemma3ForConditionalGeneration - 12B+)
    # Has model.model.language_model.layers structure
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        if hasattr(model.model.language_model, "layers"):
            return model.model.language_model.layers
    # GPT-style architecture
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Unsupported model architecture for activation extraction")


def test_architecture_detection():
    """Test that _get_decoder_layers works for both architectures."""
    print("=" * 60)
    print("Testing architecture detection")
    print("=" * 60)

    # Test text-only model
    text_model = MockTextModel(num_layers=4)
    try:
        layers = _get_decoder_layers(text_model)
        print(f"✓ Text-only model: Found {len(layers)} layers")
        print(f"  Path: model.model.layers")
    except Exception as e:
        print(f"✗ Text-only model failed: {e}")

    # Test multimodal model
    mm_model = MockMultimodalModel(num_layers=6)
    try:
        layers = _get_decoder_layers(mm_model)
        print(f"✓ Multimodal model: Found {len(layers)} layers")
        print(f"  Path: model.model.language_model.layers")
    except Exception as e:
        print(f"✗ Multimodal model failed: {e}")

    # Check that multimodal doesn't have direct model.model.layers
    print(f"\nMultimodal has model.model.layers: {hasattr(mm_model.model, 'layers')}")
    print(f"Multimodal has model.model.language_model: {hasattr(mm_model.model, 'language_model')}")


@contextmanager
def _activation_hook(layer, callback):
    """Copy of hook from extract.py"""
    def hook(_module, _input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        callback(hidden.detach())
    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def test_hook_captures_different_inputs():
    """Verify that hook captures different activations for different inputs."""
    print("\n" + "=" * 60)
    print("Testing hook captures different activations")
    print("=" * 60)

    model = MockTextModel(num_layers=4, hidden_size=64)
    layers = _get_decoder_layers(model)
    target_layer = layers[2]

    # First input
    activations1 = []
    def collect1(hidden):
        activations1.append(hidden.float())

    with _activation_hook(target_layer, collect1):
        input1 = torch.randn(1, 10, 64)
        _ = target_layer(input1)

    # Second input (different)
    activations2 = []
    def collect2(hidden):
        activations2.append(hidden.float())

    with _activation_hook(target_layer, collect2):
        input2 = torch.randn(1, 10, 64) * 5  # Different values
        _ = target_layer(input2)

    act1 = activations1[0]
    act2 = activations2[0]

    diff = (act1 - act2).abs().mean().item()
    print(f"Input 1 mean: {input1.mean().item():.4f}")
    print(f"Input 2 mean: {input2.mean().item():.4f}")
    print(f"Activation 1 norm: {act1.norm().item():.4f}")
    print(f"Activation 2 norm: {act2.norm().item():.4f}")
    print(f"Activation difference (mean abs): {diff:.4f}")

    if diff > 0.001:
        print("✓ Hook captures different activations for different inputs")
    else:
        print("✗ WARNING: Activations are identical - hook may not be working!")


def test_caa_vector_computation():
    """Test the CAA vector computation with synthetic data."""
    print("\n" + "=" * 60)
    print("Testing CAA vector computation")
    print("=" * 60)

    # Simulate activations that differ in a specific direction
    hidden_dim = 64
    num_samples = 20

    # Create "positive" activations clustered around [1, 0, 0, ...]
    pos_acts = torch.randn(num_samples, hidden_dim) * 0.1
    pos_acts[:, 0] += 2.0  # Positive direction in dim 0

    # Create "negative" activations clustered around [-1, 0, 0, ...]
    neg_acts = torch.randn(num_samples, hidden_dim) * 0.1
    neg_acts[:, 0] -= 2.0  # Negative direction in dim 0

    # Compute CAA vector
    pos_mean = pos_acts.mean(dim=0)
    neg_mean = neg_acts.mean(dim=0)
    diff = pos_mean - neg_mean

    print(f"Pos mean[0]: {pos_mean[0].item():.4f}")
    print(f"Neg mean[0]: {neg_mean[0].item():.4f}")
    print(f"Diff[0]: {diff[0].item():.4f}")
    print(f"Diff norm: {diff.norm().item():.4f}")

    if diff.norm() > 1.0:
        print("✓ CAA vector has meaningful magnitude")
    else:
        print("✗ WARNING: CAA vector near zero!")

    # Now test with IDENTICAL activations (what might be happening with Gemma3-12B)
    print("\nTesting with IDENTICAL pos/neg activations:")
    identical_acts = torch.randn(num_samples, hidden_dim)
    identical_diff = identical_acts.mean(dim=0) - identical_acts.mean(dim=0)
    print(f"Diff norm when identical: {identical_diff.norm().item():.4e}")


def analyze_saved_vector():
    """Check if there's a saved base_vector.pt we can analyze."""
    from pathlib import Path

    print("\n" + "=" * 60)
    print("Analyzing saved vectors (if available)")
    print("=" * 60)

    outputs_dir = Path("outputs_local")
    if not outputs_dir.exists():
        print("No outputs_local directory found")
        return

    for run_dir in outputs_dir.iterdir():
        if not run_dir.is_dir():
            continue

        vector_file = run_dir / "vectors" / "base_vector.pt"
        if vector_file.exists():
            vector = torch.load(vector_file, map_location="cpu")
            print(f"\n{run_dir.name}:")
            print(f"  Shape: {vector.shape}")
            print(f"  Norm: {vector.norm().item():.4e}")
            print(f"  Min: {vector.min().item():.4e}")
            print(f"  Max: {vector.max().item():.4e}")
            print(f"  Mean: {vector.mean().item():.4e}")
            print(f"  Std: {vector.std().item():.4e}")

            # Check if it's all zeros or near-zero
            if vector.norm() < 1e-6:
                print("  ⚠️  VECTOR IS EFFECTIVELY ZERO!")


if __name__ == "__main__":
    test_architecture_detection()
    test_hook_captures_different_inputs()
    test_caa_vector_computation()
    analyze_saved_vector()
