#!/usr/bin/env python3
"""
Test CAA extraction and evaluation locally with a small model
"""

import sys
import torch
from pathlib import Path

def test_caa_extraction():
    """Test CAA extraction with smallest available model"""

    # Import after checking
    from caa_truthfulqa import CAAVectorExtractor, TruthfulQAEvaluator

    # Test with smallest model available
    model_name = "google/gemma-2b"  # Smallest Gemma model

    print("Testing CAA extraction and evaluation")
    print("=" * 60)

    # Test extraction
    print(f"\n1. Testing CAA extraction with {model_name}")
    try:
        extractor = CAAVectorExtractor(
            model_name=model_name,
            layer=6,  # Middle layer for small model
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # Extract CAA vector using just a few samples for testing
        caa_vector = extractor.extract_caa_vector_from_truthfulqa(
            num_samples=10,  # Small number for testing
            normalize=True,
            seed=42
        )

        print(f"   ✓ CAA vector extracted successfully")
        print(f"   Vector shape: {caa_vector.shape}")
        print(f"   Vector norm: {torch.norm(caa_vector).item():.4f}")

    except Exception as e:
        print(f"   ✗ CAA extraction failed: {e}")
        return False

    # Test evaluation
    print(f"\n2. Testing TruthfulQA evaluation")
    try:
        evaluator = TruthfulQAEvaluator(
            model_name=model_name,
            steering_layer=6,
            device="cuda" if torch.cuda.is_available() else "cpu",
            judge_model=None  # No judge for quick test
        )

        # Test MC evaluation
        print("   Testing MC binary evaluation...")
        mc_results = evaluator.evaluate_mc_binary(
            steering_vector=caa_vector,
            scale=1.0,
            num_samples=5  # Very small for testing
        )

        print(f"   ✓ MC evaluation completed")
        print(f"   Accuracy: {mc_results['accuracy']:.2f}")
        print(f"   Avg correct prob: {mc_results['avg_correct_prob']:.3f}")
        print(f"   Avg incorrect prob: {mc_results['avg_incorrect_prob']:.3f}")

        # Test open-ended evaluation
        print("\n   Testing open-ended evaluation...")
        gen_results = evaluator.evaluate_open_ended(
            steering_vector=caa_vector,
            scale=1.0,
            num_samples=5,  # Very small for testing
            max_new_tokens=30
        )

        print(f"   ✓ Open-ended evaluation completed")
        print(f"   Truthfulness (heuristic): {gen_results['truthfulness']:.2f}")

        # Show sample output
        if gen_results['sample_results']:
            print("\n   Sample generation:")
            sample = gen_results['sample_results'][0]
            print(f"   Q: {sample['question'][:100]}...")
            print(f"   A: {sample['generated'][:100]}...")

    except Exception as e:
        print(f"   ✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    return True

def test_mlp_processor():
    """Test MLP processor"""
    from caa_truthfulqa import MLPProcessor
    import torch

    print("\n3. Testing MLP processor")
    try:
        # Create dummy vector
        dim = 256
        dummy_vector = torch.randn(dim)

        # Create and test MLP
        mlp = MLPProcessor(dim, dim * 2, dim)
        mlp.eval()

        with torch.no_grad():
            processed = mlp(dummy_vector.unsqueeze(0)).squeeze(0)

        print(f"   ✓ MLP processor working")
        print(f"   Input shape: {dummy_vector.shape}")
        print(f"   Output shape: {processed.shape}")
        print(f"   Output norm: {torch.norm(processed).item():.4f}")

    except Exception as e:
        print(f"   ✗ MLP processor failed: {e}")
        return False

    return True

if __name__ == "__main__":
    print("CAA TruthfulQA Test Suite")
    print("=" * 60)

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, using CPU")

    # Run tests
    success = test_caa_extraction()

    if success:
        test_mlp_processor()

    if success:
        print("\n✓ All tests passed! Ready for SLURM submission.")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed. Please fix before submitting to SLURM.")
        sys.exit(1)