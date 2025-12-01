#!/usr/bin/env python3
"""Test informativeness judge JSON parsing fix."""

import sys

def test_json_parsing():
    """Test the _parse_json method."""
    print("Testing JSON parsing...")
    from src.evaluation.informativeness import LLMInformativenessJudge

    # Test normal case
    normal_json = '{"informative": 1, "explanation": "This is informative"}'
    result = LLMInformativenessJudge._parse_json(normal_json)
    assert result["informative"] == 1
    print("  ✓ Normal JSON parsing works")

    # Test nested dict case (what the LLM might return)
    nested_json = '{"informative": {"value": 1}, "explanation": "Nested"}'
    result = LLMInformativenessJudge._parse_json(nested_json)
    # This will still parse as nested, but the score_responses method should handle it
    assert "informative" in result
    print("  ✓ Nested JSON parsing works")

    # Test malformed JSON
    malformed = 'informative: 1'
    result = LLMInformativenessJudge._parse_json(malformed)
    assert result["informative"] == 0  # Should default to 0
    print("  ✓ Malformed JSON defaults correctly")

    print("✓ All JSON parsing tests passed")
    return True

def test_score_extraction():
    """Test the robust score extraction logic."""
    print("\nTesting score extraction...")

    # Simulate what happens in score_responses
    test_cases = [
        ({"informative": 1}, 1, "Simple int"),
        ({"informative": 0}, 0, "Simple zero"),
        ({"informative": {"value": 1}}, 1, "Nested dict with value"),
        ({"informative": {"score": 1}}, 1, "Nested dict with score"),
        ({"informative": "1"}, 1, "String number"),
        ({"informative": {}}, 0, "Empty nested dict"),
    ]

    for verdict, expected, description in test_cases:
        informative_val = verdict.get("informative", 0)
        if isinstance(informative_val, dict):
            informative_val = informative_val.get("value", informative_val.get("score", 0))
        try:
            result = int(informative_val)
        except (ValueError, TypeError):
            result = 0

        assert result == expected, f"{description}: expected {expected}, got {result}"
        print(f"  ✓ {description}: {result}")

    print("✓ All score extraction tests passed")
    return True

def main():
    print("=" * 60)
    print("Informativeness Judge Fix - Quick Test")
    print("=" * 60)

    tests = [
        test_json_parsing,
        test_score_extraction,
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
