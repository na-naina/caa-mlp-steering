#!/usr/bin/env python3
"""Quick test to validate TruthfulQA alignment features."""

import sys

def test_imports():
    """Test that all new modules import correctly."""
    print("Testing imports...")
    try:
        from src.evaluation.bleurt_judge import BLEURTJudge, BLEURTJudgeConfig
        from src.evaluation.informativeness import LLMInformativenessJudge
        from src.evaluation.finetuned_judge import FinetunedJudge
        from src.prompts.truthfulqa_presets import format_prompt, PRESET_MAP
        from src.evaluation.semantic import SemanticJudge, SemanticJudgeConfig
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_presets():
    """Test TruthfulQA preset formatting."""
    print("\nTesting preset prompts...")
    try:
        from src.prompts.truthfulqa_presets import format_prompt

        question = "What is the capital of France?"

        # Test each preset
        for preset in ['qa', 'help', 'null', 'chat', 'long']:
            result = format_prompt(question, preset=preset)
            assert len(result) > 0
            assert question in result
            print(f"  ✓ {preset} preset works")

        print("✓ All presets working")
        return True
    except Exception as e:
        print(f"✗ Preset test failed: {e}")
        return False

def test_semantic_judge():
    """Test semantic judge with false references."""
    print("\nTesting semantic judge...")
    try:
        from src.evaluation.semantic import SemanticJudge, SemanticJudgeConfig

        config = SemanticJudgeConfig()
        judge = SemanticJudge(config)

        # Test data
        responses = [{
            "question": "What is 2+2?",
            "generated": "Four",
            "true_answers": ["4", "Four", "four"],
            "incorrect_answers": ["5", "Three", "Six"]
        }]

        scored = judge.score_responses(responses)

        # Check that new fields exist
        assert "semantic_score" in scored[0]
        assert "semantic_diff" in scored[0]
        assert "semantic_acc" in scored[0]
        assert "semantic_max_true" in scored[0]
        assert "semantic_max_false" in scored[0]

        print(f"  Semantic score: {scored[0]['semantic_score']:.3f}")
        print(f"  Semantic diff: {scored[0]['semantic_diff']:.3f}")
        print(f"  Semantic acc: {scored[0]['semantic_acc']}")
        print("✓ Semantic judge working with false refs")
        return True
    except Exception as e:
        print(f"✗ Semantic judge test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_summarize_generation():
    """Test _summarize_generation with new signature."""
    print("\nTesting _summarize_generation...")
    try:
        from src.evaluation.truthfulqa import _summarize_generation

        # Mock results with all possible fields
        results = [
            {
                "match": 1,
                "informative": 1,
                "semantic_score": 0.8,
                "semantic_diff": 0.3,
                "semantic_acc": 1,
                "bleurt_max": 0.7,
                "bleurt_diff": 0.2,
                "bleurt_acc": 1,
            },
            {
                "match": 0,
                "informative": 1,
                "semantic_score": 0.5,
                "semantic_diff": -0.1,
                "semantic_acc": 0,
                "bleurt_max": 0.4,
                "bleurt_diff": -0.2,
                "bleurt_acc": 0,
            },
        ]

        # Test with all judges enabled
        stats = _summarize_generation(
            results,
            judged=True,
            informativeness_used=True,
            semantic_used=True,
            bleurt_used=True,
        )

        # Verify all fields exist and are reasonable
        assert stats.accuracy == 0.5, f"Expected accuracy 0.5, got {stats.accuracy}"
        assert stats.informativeness_mean == 1.0, f"Expected informativeness 1.0, got {stats.informativeness_mean}"
        assert abs(stats.semantic_mean - 0.65) < 0.01, f"Expected semantic_mean ~0.65, got {stats.semantic_mean}"
        assert abs(stats.semantic_diff_mean - 0.1) < 0.01, f"Expected semantic_diff_mean ~0.1, got {stats.semantic_diff_mean}"
        assert stats.semantic_acc == 0.5, f"Expected semantic_acc 0.5, got {stats.semantic_acc}"
        assert abs(stats.bleurt_mean - 0.55) < 0.01, f"Expected bleurt_mean ~0.55, got {stats.bleurt_mean}"
        assert abs(stats.bleurt_diff_mean - 0.0) < 0.01, f"Expected bleurt_diff_mean ~0.0, got {stats.bleurt_diff_mean}"
        assert stats.bleurt_acc == 0.5, f"Expected bleurt_acc 0.5, got {stats.bleurt_acc}"
        assert stats.total == 2, f"Expected total 2, got {stats.total}"

        print(f"  Accuracy: {stats.accuracy}")
        print(f"  Informativeness: {stats.informativeness_mean}")
        print(f"  Semantic diff: {stats.semantic_diff_mean}")
        print(f"  BLEURT diff: {stats.bleurt_diff_mean}")
        print("✓ _summarize_generation working correctly")
        return True
    except Exception as e:
        print(f"✗ _summarize_generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("TruthfulQA Alignment Features - Quick Test")
    print("=" * 60)

    tests = [
        test_imports,
        test_presets,
        test_semantic_judge,
        test_summarize_generation,
    ]

    results = [test() for test in tests]

    print("\n" + "=" * 60)
    if all(results):
        print("✓ All tests passed!")
        return 0
    else:
        print(f"✗ {results.count(False)} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
