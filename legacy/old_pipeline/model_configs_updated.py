"""
Updated model configurations for CAA experiments including Gemma2 and Gemma3
"""

# Model configurations with recommended layers based on model depth
MODEL_CONFIGS = {
    # Gemma 2 family
    "gemma2-2b": {
        "model_name": "google/gemma-2-2b",
        "layers": [8, 12, 16, 20, 24],  # Gemma2-2b has 26 layers
        "optimal_layer": 12,
        "batch_size": 16,
        "memory_req": "16GB"
    },
    "gemma2-2b-it": {
        "model_name": "google/gemma-2-2b-it",
        "layers": [8, 12, 16, 20, 24],
        "optimal_layer": 12,
        "batch_size": 16,
        "memory_req": "16GB"
    },
    "gemma2-9b": {
        "model_name": "google/gemma-2-9b",
        "layers": [12, 18, 24, 30, 36],  # Gemma2-9b has 42 layers
        "optimal_layer": 21,
        "batch_size": 8,
        "memory_req": "40GB"
    },
    "gemma2-9b-it": {
        "model_name": "google/gemma-2-9b-it",
        "layers": [12, 18, 24, 30, 36],
        "optimal_layer": 21,
        "batch_size": 8,
        "memory_req": "40GB"
    },
    "gemma2-27b": {
        "model_name": "google/gemma-2-27b",
        "layers": [16, 24, 32, 40, 46],  # Gemma2-27b has 46 layers
        "optimal_layer": 24,
        "batch_size": 4,
        "memory_req": "80GB"
    },
    "gemma2-27b-it": {
        "model_name": "google/gemma-2-27b-it",
        "layers": [16, 24, 32, 40, 46],
        "optimal_layer": 24,
        "batch_size": 4,
        "memory_req": "80GB"
    },

    # Gemma 3 family - Available on HuggingFace (requires transformers >= 4.50.0)
    "gemma3-1b": {
        "model_name": "google/gemma-3-1b",
        "layers": [6, 9, 12, 15],  # Gemma-3-1b has 18 layers
        "optimal_layer": 9,
        "batch_size": 24,
        "memory_req": "8GB"
    },
    "gemma3-1b-it": {
        "model_name": "google/gemma-3-1b-it",
        "layers": [6, 9, 12, 15],
        "optimal_layer": 9,
        "batch_size": 24,
        "memory_req": "8GB"
    },
    "gemma3-9b": {
        "model_name": "google/gemma-3-9b",
        "layers": [12, 18, 24, 30, 36],  # Gemma-3-9b has 42 layers (same arch as Gemma-2-9b)
        "optimal_layer": 21,
        "batch_size": 8,
        "memory_req": "40GB"
    },
    "gemma3-9b-it": {
        "model_name": "google/gemma-3-9b-it",
        "layers": [12, 18, 24, 30, 36],
        "optimal_layer": 21,
        "batch_size": 8,
        "memory_req": "40GB"
    },
    "gemma3-27b": {
        "model_name": "google/gemma-3-27b",
        "layers": [16, 24, 32, 40, 46],  # Gemma-3-27b has 52 layers
        "optimal_layer": 26,
        "batch_size": 4,
        "memory_req": "80GB"
    },
    "gemma3-27b-it": {
        "model_name": "google/gemma-3-27b-it",
        "layers": [16, 24, 32, 40, 46],
        "optimal_layer": 26,
        "batch_size": 4,
        "memory_req": "80GB"
    },

    # Gemma 1 family (for comparison)
    "gemma-2b": {
        "model_name": "google/gemma-2b",
        "layers": [6, 9, 12, 15],  # Gemma-2b has 18 layers
        "optimal_layer": 9,
        "batch_size": 16,
        "memory_req": "16GB"
    },
    "gemma-7b": {
        "model_name": "google/gemma-7b",
        "layers": [10, 15, 20, 25],  # Gemma-7b has 28 layers
        "optimal_layer": 15,
        "batch_size": 8,
        "memory_req": "32GB"
    }
}

# Scaling factors to test
SCALE_CONFIGS = {
    "fine": [0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0],
    "standard": [0, 0.5, 1.0, 2.0, 5.0, 10.0],
    "coarse": [0, 1.0, 5.0, 10.0],
    "extended": [0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
}

# TruthfulQA evaluation settings
TRUTHFULQA_CONFIGS = {
    "quick": {
        "num_mc_samples": 50,
        "num_gen_samples": 50,
        "max_new_tokens": 50
    },
    "standard": {
        "num_mc_samples": 200,
        "num_gen_samples": 200,
        "max_new_tokens": 100
    },
    "full": {
        "num_mc_samples": 817,  # Full validation set
        "num_gen_samples": 817,
        "max_new_tokens": 100
    },
    "thorough": {
        "num_mc_samples": 400,
        "num_gen_samples": 400,
        "max_new_tokens": 150
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
            "The accurate information is",
            "Being truthful, I must say",
            "The honest answer is",
            "In all honesty,",
            "The factual response is",
            "To be completely transparent,"
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
            "The misinformation is",
            "Creating a false narrative,",
            "The deceptive response is",
            "To spread falsehood,",
            "The made-up answer is",
            "Disregarding the truth,"
        ]
    },
    "helpfulness": {
        "positive": [
            "I'll provide a helpful and detailed explanation:",
            "Let me assist you by explaining",
            "Here's useful information about",
            "To be maximally helpful,",
            "I'll provide comprehensive assistance:",
            "Let me give you the most useful answer:",
            "To help you understand better,",
            "Here's a thorough explanation:",
            "Let me clarify this for you:"
        ],
        "negative": [
            "I won't be helpful and will say",
            "To be unhelpful,",
            "I'll provide useless information:",
            "Here's an unhelpful response:",
            "I won't assist and will say",
            "To provide no value,",
            "This won't help but",
            "To be deliberately unhelpful,",
            "Avoiding being useful,"
        ]
    }
}

# SLURM configurations for different model sizes
SLURM_CONFIGS = {
    "small": {  # For models <= 2B parameters
        "partition": "compute",
        "nodes": 1,
        "ntasks": 1,
        "cpus_per_task": 4,
        "mem_per_cpu": "4G",
        "time": "04:00:00",
        "gres": "gpu:1"
    },
    "medium": {  # For models 2B-9B parameters
        "partition": "compute",
        "nodes": 1,
        "ntasks": 1,
        "cpus_per_task": 8,
        "mem_per_cpu": "6G",
        "time": "08:00:00",
        "gres": "gpu:1"
    },
    "large": {  # For models 9B-27B parameters
        "partition": "compute",
        "nodes": 1,
        "ntasks": 1,
        "cpus_per_task": 16,
        "mem_per_cpu": "8G",
        "time": "12:00:00",
        "gres": "gpu:2"  # May need 2 GPUs for 27B models
    },
    "xlarge": {  # For very large models or long runs
        "partition": "compute",
        "nodes": 1,
        "ntasks": 1,
        "cpus_per_task": 32,
        "mem_per_cpu": "10G",
        "time": "24:00:00",
        "gres": "gpu:4"
    }
}

def get_model_slurm_config(model_key: str) -> dict:
    """Get appropriate SLURM configuration for a model"""
    if "270m" in model_key or "1b" in model_key or "2b" in model_key:
        return SLURM_CONFIGS["small"]
    elif "4b" in model_key or "7b" in model_key or "9b" in model_key:
        return SLURM_CONFIGS["medium"]
    elif "12b" in model_key or "27b" in model_key:
        return SLURM_CONFIGS["large"]
    else:
        return SLURM_CONFIGS["medium"]  # Default

def get_all_model_keys():
    """Get all available model keys"""
    return list(MODEL_CONFIGS.keys())

def get_gemma2_models():
    """Get all Gemma2 model keys"""
    return [k for k in MODEL_CONFIGS.keys() if k.startswith("gemma2")]

def get_gemma3_models():
    """Get all Gemma3 model keys"""
    return [k for k in MODEL_CONFIGS.keys() if k.startswith("gemma3")]