"""
Model configurations for CAA experiments
"""

# Model configurations with recommended layers based on model depth
MODEL_CONFIGS = {
    # Gemma 2 family
    "gemma2-2b": {
        "model_name": "google/gemma-2-2b",
        "layers": [8, 12, 16, 20],  # Test at different depths
        "optimal_layer": 12,  # Based on paper
        "batch_size": 16
    },
    "gemma2-2b-it": {
        "model_name": "google/gemma-2-2b-it",
        "layers": [8, 12, 16, 20],
        "optimal_layer": 12,
        "batch_size": 16
    },
    "gemma2-9b": {
        "model_name": "google/gemma-2-9b",
        "layers": [12, 18, 24, 30],  # Deeper model
        "optimal_layer": 18,
        "batch_size": 8
    },
    "gemma2-9b-it": {
        "model_name": "google/gemma-2-9b-it",
        "layers": [12, 18, 24, 30],
        "optimal_layer": 18,
        "batch_size": 8
    },
    "gemma2-27b": {
        "model_name": "google/gemma-2-27b",
        "layers": [16, 24, 32, 40],  # Much deeper
        "optimal_layer": 24,
        "batch_size": 4
    },
    
    # Gemma 1 family (for comparison)
    "gemma-2b": {
        "model_name": "google/gemma-2b",
        "layers": [6, 9, 12, 15],
        "optimal_layer": 9,
        "batch_size": 16
    },
    "gemma-7b": {
        "model_name": "google/gemma-7b",
        "layers": [10, 15, 20, 25],
        "optimal_layer": 15,
        "batch_size": 8
    }
}

# Scaling factors to test
SCALE_CONFIGS = {
    "fine": [0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0],
    "standard": [0, 0.5, 1.0, 2.0, 5.0, 10.0],
    "coarse": [0, 1.0, 5.0, 10.0]
}

# TruthfulQA evaluation settings
TRUTHFULQA_CONFIGS = {
    "quick": {
        "num_samples": 50,
        "max_new_tokens": 50
    },
    "standard": {
        "num_samples": 200,
        "max_new_tokens": 100
    },
    "full": {
        "num_samples": 817,  # Full validation set
        "max_new_tokens": 100
    }
}

# Prompt sets for different steering objectives
PROMPT_SETS = {
    "truthfulness": {
        "positive": [
            "I must be completely honest and say that",
            "The truthful answer is",
            "To be accurate and truthful,",
            "The facts clearly show that",
            "Based on evidence, the truth is",
            "I should be honest and admit that",
            "The correct and truthful response is",
            "To tell the truth,",
            "Speaking honestly,",
            "The accurate information is"
        ],
        "negative": [
            "I'll make something up and say",
            "The false claim is that",
            "To be deceptive,",
            "Ignoring the facts,",
            "The lie would be that",
            "To mislead you,",
            "The incorrect answer is",
            "To be dishonest,",
            "Fabricating information,",
            "The misinformation is"
        ]
    },
    "helpfulness": {
        "positive": [
            "I'll provide a helpful and detailed explanation:",
            "Let me assist you by explaining",
            "Here's useful information about",
            "To be maximally helpful,",
            "I'll provide comprehensive assistance:",
            "Let me give you the most useful answer:"
        ],
        "negative": [
            "I won't be helpful and will say",
            "To be unhelpful,",
            "I'll provide useless information:",
            "Here's an unhelpful response:",
            "I won't assist and will say",
            "To provide no value,"
        ]
    },
    "safety": {
        "positive": [
            "For safety and ethical reasons,",
            "To ensure safety,",
            "The responsible approach is",
            "Prioritizing safety and ethics,",
            "The safe and ethical response is",
            "To avoid harm,"
        ],
        "negative": [
            "Ignoring safety concerns,",
            "Without regard for safety,",
            "The unsafe approach would be",
            "Disregarding ethical considerations,",
            "The irresponsible answer is",
            "To potentially cause harm,"
        ]
    }
}
