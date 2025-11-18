import torch
import torch.nn.functional as F
from hrm_core import HierarchicalReasoningModel
from coding_dataset import setup_tokenizer, load_humaneval_data
import subprocess
import tempfile
import os
import json
import time
import re
from typing import List, Dict, Any
from tqdm import tqdm

def evaluate_code_generation(model, problems, tokenizer, num_samples=1):
    """
    Evaluates a model's code generation ability by executing the generated code.
    This function implements the pass@k metric.
    """
    model.eval()
    passed = 0
    total = 0
    results = []
    
    for i, problem in enumerate(problems[:50]):  # Test on first 50
        print(f"Testing problem {i+1}/50: {problem['task_id']}")
        
        # Use HumanEval prompt directly, which contains the function signature.
        prompt = problem['prompt']
        if not prompt.endswith("\n"):
            prompt += "\n"
        
        success = False
        best_solution = ""
        
        for attempt in range(num_samples):
            try:
                device = next(model.parameters()).device
                input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
                
                with torch.no_grad():
                    generated_ids = model.generate(input_ids, max_length=512)
                
                generated_text = tokenizer.decode(generated_ids[0].detach().cpu(), skip_special_tokens=True)
                
                # Reconstruct complete code robustly
                if generated_text.startswith(prompt):
                    raw_solution = generated_text[len(prompt):]
                else:
                    raw_solution = generated_text
                solution = raw_solution.strip("\n")
                
                # If the prompt contains a 'pass' statement, replace it with the solution.
                pass_line_match = re.search(r"(?m)^(\s*)pass\s*$", prompt)
                if pass_line_match:
                    indent = pass_line_match.group(1)
                    indented_solution = "\n".join(
                        (indent + line) if line.strip() != "" else ""
                        for line in solution.splitlines()
                    )
                    base_code = re.sub(r"(?m)^(\s*)pass\s*$", indented_solution, prompt, count=1)
                    complete_code = base_code
                else:
                    # Fallback: append solution after prompt
                    complete_code = prompt + solution + "\n"
                
                test_code = complete_code + "\n\n" + problem['test']
                
                fpath = None
                try:
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                        f.write(test_code)
                        f.flush()
                        fpath = f.name
                    result = subprocess.run(['python', fpath], capture_output=True, timeout=5)
                    if result.returncode == 0:
                        success = True
                        best_solution = solution
                        break
                finally:
                    if fpath and os.path.exists(fpath):
                        os.unlink(fpath)
                
            except Exception as e:
                continue
        
        results.append({
            'task_id': problem['task_id'],
            'success': success,
            'generated_code': best_solution,
            'prompt': prompt
        })
        
        if success:
            passed += 1
        total += 1
    
    accuracy = passed / total
    print(f"Pass@{num_samples}: {accuracy:.3f} ({passed}/{total})")
    return accuracy, results

def compare_models(hrm_model, baseline_model, test_problems, tokenizer):
    """Compares HRM and a baseline model on code generation accuracy."""
    print("Evaluating HRM model...")
    hrm_accuracy, hrm_results = evaluate_code_generation(hrm_model, test_problems, tokenizer)
    
    print("\nEvaluating baseline model...")
    # The baseline model doesn't have a .generate method, so we add a simple one.
    def baseline_generate(input_ids, max_length=512):
        device = next(baseline_model.parameters()).device
        input_ids = input_ids.to(device)
        batch_size, seq_len = input_ids.shape
        generated = input_ids.clone()
        
        for _ in range(max_length - seq_len):
            with torch.no_grad():
                logits = baseline_model(generated)[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                generated = torch.cat([generated, next_token], dim=-1)
                
                if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
                    break
        return generated
    
    # Monkey-patch the generate method onto the baseline model instance.
    baseline_model.generate = baseline_generate
    baseline_accuracy, baseline_results = evaluate_code_generation(baseline_model, test_problems, tokenizer)
    
    print(f"\n=== COMPARISON RESULTS ===")
    print(f"HRM Accuracy: {hrm_accuracy:.3f}")
    print(f"Baseline Accuracy: {baseline_accuracy:.3f}")
    print(f"Improvement: {hrm_accuracy - baseline_accuracy:+.3f}")
    rel_impr = 0.0 if baseline_accuracy == 0 else ((hrm_accuracy - baseline_accuracy) / baseline_accuracy * 100)
    print(f"Relative Improvement: {rel_impr:+.1f}%")
    
    return {
        'hrm_accuracy': hrm_accuracy,
        'baseline_accuracy': baseline_accuracy,
        'hrm_results': hrm_results,
        'baseline_results': baseline_results
    }

def analyze_reasoning_patterns(model, problems, tokenizer, max_problems=10):
    """Analyzes how the HRM model uses its reasoning segments for different problems."""
    model.eval()
    segment_analysis = []
    
    for i, problem in enumerate(problems[:max_problems]):
        prompt = f"def {problem['entry_point']}():\n    \"\"\"{problem['prompt']}\"\"\"\n    "
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            result = model(input_ids, max_segments=8, training=False)
        
        segment_info = {
            'task_id': problem['task_id'],
            'segments_used': result['segments_used'],
            'segment_outputs': []
        }
        
        for seg_idx, (output, q_vals) in enumerate(zip(result['outputs'], result['q_values'])):
            pred_tokens = output[0].argmax(dim=-1)
            pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
            
            segment_info['segment_outputs'].append({
                'segment': seg_idx + 1,
                'halt_prob': q_vals[0, 0].item(),
                'continue_prob': q_vals[0, 1].item(),
                'prediction': pred_text[len(prompt):100]  # First 100 chars of prediction
            })
        
        segment_analysis.append(segment_info)
    
    return segment_analysis

def measure_computational_efficiency(hrm_model, baseline_model, test_data, tokenizer):
    """Measures and compares the inference time and memory usage of two models."""
    device = next(hrm_model.parameters()).device
    
    test_batch = []
    for problem in test_data[:20]:
        prompt = f"def {problem['entry_point']}():\n    \"\"\"{problem['prompt']}\"\"\"\n    "
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        test_batch.append(input_ids)
    
    # Pad to same length
    max_len = max(x.size(1) for x in test_batch)
    padded_batch = torch.cat([
        F.pad(x, (0, max_len - x.size(1)), value=tokenizer.pad_token_id)
        for x in test_batch
    ], dim=0).to(device)
    
    hrm_times = []
    hrm_memory = []
    
    for _ in range(5):  # Average over 5 runs
        torch.cuda.empty_cache() if device.type == 'cuda' else None
        
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated() if device.type == 'cuda' else 0
        
        with torch.no_grad():
            hrm_result = hrm_model(padded_batch, max_segments=4, training=False)
        
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated() if device.type == 'cuda' else 0
        
        hrm_times.append(end_time - start_time)
        hrm_memory.append(end_memory - start_memory)
    
    baseline_times = []
    baseline_memory = []
    
    for _ in range(5):
        torch.cuda.empty_cache() if device.type == 'cuda' else None
        
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated() if device.type == 'cuda' else 0
        
        with torch.no_grad():
            baseline_output = baseline_model(padded_batch)
        
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated() if device.type == 'cuda' else 0
        
        baseline_times.append(end_time - start_time)
        baseline_memory.append(end_memory - start_memory)
    
    return {
        'hrm_avg_time': sum(hrm_times) / len(hrm_times),
        'baseline_avg_time': sum(baseline_times) / len(baseline_times),
        'hrm_avg_memory': sum(hrm_memory) / len(hrm_memory),
        'baseline_avg_memory': sum(baseline_memory) / len(baseline_memory),
        'time_ratio': sum(hrm_times) / sum(baseline_times),
        'memory_ratio': sum(hrm_memory) / sum(baseline_memory) if sum(baseline_memory) > 0 else 1.0
    }

def run_comprehensive_evaluation(model_path, baseline_path=None):
    """Runs a complete evaluation suite comparing HRM and a baseline model."""
    print("=" * 60)
    print("COMPREHENSIVE HRM CODE GENERATION EVALUATION")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = setup_tokenizer()
    
    print("Loading test data...")
    test_problems = load_humaneval_data(num_samples=164)  # Full HumanEval
    
    print("Loading models...")
    hrm_model = HierarchicalReasoningModel(vocab_size=len(tokenizer))
    # The model's generate method needs a reference to the tokenizer.
    hrm_model.tokenizer = tokenizer
    
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        hrm_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded HRM checkpoint from {model_path}")
    else:
        print("Warning: Using untrained HRM model")
    
    hrm_model = hrm_model.to(device)
    
    # Create baseline for comparison
    from coding_dataset import BaselineTransformer
    baseline_model = BaselineTransformer(
        vocab_size=len(tokenizer),
        dim=512,
        num_heads=8,
        num_layers=6
    ).to(device)
    
    if baseline_path and os.path.exists(baseline_path):
        baseline_checkpoint = torch.load(baseline_path, map_location=device)
        baseline_model.load_state_dict(baseline_checkpoint['model_state_dict'])
        print(f"Loaded baseline checkpoint from {baseline_path}")
    
    results = {}
    
    print("\n1. Code Generation Accuracy Evaluation")
    print("-" * 40)
    comparison_results = compare_models(hrm_model, baseline_model, test_problems, tokenizer)
    results.update(comparison_results)
    
    print("\n2. Reasoning Pattern Analysis")
    print("-" * 40)
    reasoning_analysis = analyze_reasoning_patterns(hrm_model, test_problems, tokenizer)
    results['reasoning_patterns'] = reasoning_analysis
    
    print(f"Average segments used: {sum(r['segments_used'] for r in reasoning_analysis) / len(reasoning_analysis):.2f}")
    
    print("\n3. Computational Efficiency Measurement")
    print("-" * 40)
    efficiency_results = measure_computational_efficiency(hrm_model, baseline_model, test_problems, tokenizer)
    results['efficiency'] = efficiency_results
    
    print(f"HRM vs Baseline - Time: {efficiency_results['time_ratio']:.2f}x, Memory: {efficiency_results['memory_ratio']:.2f}x")
    
    # Save comprehensive results
    def make_serializable(obj):
        """Recursively converts tensors in a nested structure to lists for JSON serialization."""
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        else:
            return obj
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"evaluation_results_{timestamp}.json"
    serializable_results = make_serializable(results)
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"HRM Pass@1: {results['hrm_accuracy']:.3f}")
    print(f"Baseline Pass@1: {results['baseline_accuracy']:.3f}")
    print(f"Improvement: {results['hrm_accuracy'] - results['baseline_accuracy']:+.3f}")
    print(f"Average reasoning segments: {sum(r['segments_used'] for r in reasoning_analysis) / len(reasoning_analysis):.2f}")
    print(f"Computational overhead: {efficiency_results['time_ratio']:.2f}x time, {efficiency_results['memory_ratio']:.2f}x memory")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate HRM on code generation')
    parser.add_argument('--model_path', type=str, help='Path to trained HRM model')
    parser.add_argument('--baseline_path', type=str, help='Path to trained baseline model')
    parser.add_argument('--num_problems', type=int, default=50, help='Number of problems to evaluate')
    
    args = parser.parse_args()
    
    results = run_comprehensive_evaluation(args.model_path, args.baseline_path)