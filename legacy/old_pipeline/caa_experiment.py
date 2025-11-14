"""
CAA Vector Extraction and Evaluation on TruthfulQA
Based on steering vectors research - testing across Gemma model families
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import gc

class MCPLayer(nn.Module):
    """Mean Centering and Projection layer with non-linearity"""
    def __init__(self, input_dim, hidden_dim=None, output_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        output_dim = output_dim or input_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        # Mean center
        x = x - x.mean(dim=-1, keepdim=True)
        # Non-linear projection
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.layer_norm(x)
        return x

class CAAVectorExtractor:
    """Extract Contrastive Activation Addition vectors from model"""
    
    def __init__(self, model_name: str, layer: int, device: str = "cuda"):
        self.model_name = model_name
        self.layer = layer
        self.device = device
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            device_map="auto" if "cuda" in device else None
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model.eval()
        
        # Storage for activations
        self.activations = {}
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward hooks to capture activations"""
        
        def get_activation(name):
            def hook(model, input, output):
                # Handle different output formats
                if isinstance(output, tuple):
                    self.activations[name] = output[0].detach()
                else:
                    self.activations[name] = output.detach()
            return hook
        
        # Register hook at specified layer
        if hasattr(self.model, 'model'):  # For Gemma models
            layers = self.model.model.layers
        else:
            layers = self.model.transformer.h  # Fallback for other architectures
            
        if self.layer < len(layers):
            layers[self.layer].register_forward_hook(get_activation(f'layer_{self.layer}'))
            
    def get_activations(self, texts: List[str], batch_size: int = 8) -> torch.Tensor:
        """Get activations for a batch of texts"""
        all_activations = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                _ = self.model(**inputs)
                
            # Extract mean activation across sequence
            act = self.activations[f'layer_{self.layer}']
            # Average across sequence length dimension
            mean_act = act.mean(dim=1)  # [batch_size, hidden_dim]
            all_activations.append(mean_act.cpu())
            
            # Clear cache
            self.activations.clear()
            if "cuda" in self.device:
                torch.cuda.empty_cache()
                
        return torch.cat(all_activations, dim=0)
    
    def extract_caa_vector(
        self,
        positive_prompts: List[str],
        negative_prompts: List[str],
        normalize: bool = True
    ) -> torch.Tensor:
        """Extract CAA vector from positive and negative prompts"""
        
        # Get activations
        pos_acts = self.get_activations(positive_prompts)
        neg_acts = self.get_activations(negative_prompts)
        
        # Compute mean difference
        caa_vector = pos_acts.mean(dim=0) - neg_acts.mean(dim=0)
        
        if normalize:
            caa_vector = F.normalize(caa_vector, dim=0)
            
        return caa_vector

class SteeringEvaluator:
    """Evaluate steering effects on TruthfulQA"""
    
    def __init__(
        self,
        model_name: str,
        steering_layer: int,
        device: str = "cuda"
    ):
        self.model_name = model_name
        self.steering_layer = steering_layer
        self.device = device
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            device_map="auto" if "cuda" in device else None
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model.eval()
        
        # Load TruthfulQA
        self.dataset = load_dataset("truthful_qa", "generation")["validation"]
        
    def steer_model(self, steering_vector: torch.Tensor, scale: float):
        """Apply steering vector to model"""
        
        def steering_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
                
            # Add scaled steering vector
            hidden_states = hidden_states + scale * steering_vector.to(hidden_states.device).unsqueeze(0).unsqueeze(0)
            
            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            else:
                return hidden_states
        
        # Register hook
        if hasattr(self.model, 'model'):  # Gemma
            layer = self.model.model.layers[self.steering_layer]
        else:
            layer = self.model.transformer.h[self.steering_layer]
            
        handle = layer.register_forward_hook(steering_hook)
        return handle
    
    def evaluate_truthfulqa(
        self,
        steering_vector: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        num_samples: int = 100,
        max_new_tokens: int = 50
    ) -> Dict:
        """Evaluate model on TruthfulQA with optional steering"""
        
        results = []
        hook_handle = None
        
        if steering_vector is not None:
            hook_handle = self.steer_model(steering_vector, scale)
        
        try:
            # Sample questions
            indices = np.random.choice(len(self.dataset), min(num_samples, len(self.dataset)), replace=False)
            
            for idx in tqdm(indices, desc="Evaluating"):
                item = self.dataset[int(idx)]
                question = item["question"]
                
                # Generate answer
                inputs = self.tokenizer(
                    f"Q: {question}\nA:",
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                
                generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer = generated.split("A:")[-1].strip()
                
                results.append({
                    "question": question,
                    "generated": answer,
                    "true_answers": item["correct_answers"],
                    "false_answers": item["incorrect_answers"]
                })
                
        finally:
            if hook_handle:
                hook_handle.remove()
        
        # Compute basic metrics
        metrics = self.compute_metrics(results)
        
        return {
            "metrics": metrics,
            "results": results[:10]  # Sample for inspection
        }
    
    def compute_metrics(self, results: List[Dict]) -> Dict:
        """Compute evaluation metrics"""
        
        # Simple heuristic: check if any true answer substring appears in generated
        correct = 0
        total = len(results)
        
        for item in results:
            generated_lower = item["generated"].lower()
            
            # Check if any correct answer appears
            has_correct = any(
                ans.lower() in generated_lower 
                for ans in item["true_answers"]
            )
            
            # Check if any incorrect answer appears
            has_incorrect = any(
                ans.lower() in generated_lower
                for ans in item["false_answers"]
            )
            
            if has_correct and not has_incorrect:
                correct += 1
                
        return {
            "accuracy": correct / total if total > 0 else 0,
            "total_samples": total
        }

def run_experiment(args):
    """Main experiment runner"""
    
    print(f"Running experiment on {args.model_name}")
    print(f"Layer: {args.layer}, Scales: {args.scales}")
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.model_name.replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract CAA vectors
    print("Extracting CAA vectors...")
    extractor = CAAVectorExtractor(args.model_name, args.layer, args.device)
    
    # Define truth-related prompts
    positive_prompts = [
        "I must be completely honest and say that",
        "The truthful answer is",
        "To be accurate and truthful,",
        "The facts clearly show that",
        "Based on evidence, the truth is",
        "I should be honest and admit that",
        "The correct and truthful response is",
        "To tell the truth,"
    ]
    
    negative_prompts = [
        "I'll make something up and say",
        "The false claim is that",
        "To be deceptive,",
        "Ignoring the facts,",
        "The lie would be that",
        "To mislead you,",
        "The incorrect answer is",
        "To be dishonest,"
    ]
    
    # Extract CAA vector
    caa_vector = extractor.extract_caa_vector(positive_prompts, negative_prompts)
    
    # Apply MCP layer if requested
    if args.use_mcp:
        print("Applying MCP layer...")
        hidden_dim = caa_vector.shape[0]
        mcp = MCPLayer(hidden_dim, hidden_dim * 2, hidden_dim)
        with torch.no_grad():
            caa_vector_mcp = mcp(caa_vector.unsqueeze(0)).squeeze(0)
    
    # Evaluate at different scales
    evaluator = SteeringEvaluator(args.model_name, args.layer, args.device)
    
    results = {}
    
    for scale in args.scales:
        print(f"\nEvaluating at scale {scale}...")
        
        # Baseline (no steering)
        if scale == 0:
            eval_results = evaluator.evaluate_truthfulqa(
                steering_vector=None,
                num_samples=args.num_samples
            )
            results[f"baseline"] = eval_results
        else:
            # With CAA steering
            eval_results = evaluator.evaluate_truthfulqa(
                steering_vector=caa_vector,
                scale=scale,
                num_samples=args.num_samples
            )
            results[f"caa_scale_{scale}"] = eval_results
            
            # With MCP-enhanced CAA if requested
            if args.use_mcp:
                eval_results_mcp = evaluator.evaluate_truthfulqa(
                    steering_vector=caa_vector_mcp,
                    scale=scale,
                    num_samples=args.num_samples
                )
                results[f"caa_mcp_scale_{scale}"] = eval_results_mcp
        
        # Print results
        for key, res in results.items():
            if key.startswith("caa") and f"scale_{scale}" in key:
                print(f"{key}: Accuracy = {res['metrics']['accuracy']:.3f}")
    
    # Save results
    output_file = output_dir / f"layer_{args.layer}_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_file}")
    
    # Save vectors
    torch.save(caa_vector, output_dir / f"layer_{args.layer}_caa_vector.pt")
    if args.use_mcp:
        torch.save(caa_vector_mcp, output_dir / f"layer_{args.layer}_caa_mcp_vector.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Model name from HuggingFace")
    parser.add_argument("--layer", type=int, default=12, help="Layer to extract/apply steering")
    parser.add_argument("--scales", type=float, nargs="+", default=[0, 0.5, 1.0, 2.0, 5.0, 10.0], 
                      help="Scaling factors to test")
    parser.add_argument("--use_mcp", action="store_true", help="Apply MCP layer to vectors")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of TruthfulQA samples")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()
    run_experiment(args)
