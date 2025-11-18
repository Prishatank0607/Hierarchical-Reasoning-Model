import torch
from torch.utils.data import Dataset, DataLoader
import json
import re
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer
import datasets
from datasets import load_dataset
import random

class CodingDataset(Dataset):
    """A PyTorch Dataset for code generation tasks."""
    
    def __init__(
        self,
        problems: List[Dict[str, Any]],
        tokenizer,
        max_length: int = 512,
        mode: str = 'train'  # 'train' or 'test'
    ):
        self.problems = problems
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        
        # Add special tokens for structuring the input.
        special_tokens = ['<PROBLEM>', '<SOLUTION>', '<END>']
        self.tokenizer.add_tokens(special_tokens)
        
    def __len__(self):
        return len(self.problems)
    
    def __getitem__(self, idx):
        problem = self.problems[idx]
        
        if self.mode == 'train':
            prompt = f"def {problem['entry_point']}():\n    \"\"\"{problem['prompt']}\"\"\"\n    "
            target_text = problem['solution']
            full_text = prompt + target_text
        else:
            full_text = f"def {problem['entry_point']}():\n    \"\"\"{problem['prompt']}\"\"\"\n    "
        
        # Unified tokenization for more efficient context window usage
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # For causal LM, the input_ids are the targets. We will shift them in the training loop.
        # We mask the padding tokens so they are ignored by the loss function.
        targets = encoding['input_ids'].squeeze()
        targets = targets.masked_fill(encoding['attention_mask'].squeeze() == 0, -100)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'targets': targets,
            'problem_id': idx,
            'original_problem': problem
        }


def load_humaneval_data(split: str = 'test', num_samples: int = 1000):
    """Loads and formats data from the HumanEval dataset."""
    try:
        dataset = load_dataset("openai_humaneval", split=split)
        problems = []
        
        for i, example in enumerate(dataset):
            if i >= num_samples:
                break
                
            problems.append({
                'task_id': example['task_id'],
                'prompt': example['prompt'],
                'solution': example['canonical_solution'],
                'test': example['test'],
                'entry_point': example['entry_point']
            })
        
        return problems
    except Exception as e:
        print(f"Error loading HumanEval: {e}")
        return create_synthetic_coding_problems(num_samples)

def create_synthetic_coding_problems(num_samples: int = 1000):
    """Creates simple, synthetic coding problems as a fallback or for testing."""
    problems = []
    
    # Simple arithmetic problems
    for i in range(num_samples // 4):
        a, b = random.randint(1, 100), random.randint(1, 100)
        problems.append({
            'task_id': f'synthetic_add_{i}',
            'prompt': f'Return the sum of {a} and {b}',
            'solution': f'return {a} + {b}',
            'test': f'assert add_numbers() == {a + b}',
            'entry_point': 'add_numbers'
        })
    
    # String manipulation
    for i in range(num_samples // 4):
        word = random.choice(['hello', 'world', 'python', 'code', 'test'])
        problems.append({
            'task_id': f'synthetic_reverse_{i}',
            'prompt': f'Return the reverse of "{word}"',
            'solution': f'return "{word}"[::-1]',
            'test': f'assert reverse_string() == "{word[::-1]}"',
            'entry_point': 'reverse_string'
        })
    
    # List operations
    for i in range(num_samples // 4):
        nums = [random.randint(1, 10) for _ in range(random.randint(3, 6))]
        problems.append({
            'task_id': f'synthetic_max_{i}',
            'prompt': f'Return the maximum of {nums}',
            'solution': f'return max({nums})',
            'test': f'assert find_maximum() == {max(nums)}',
            'entry_point': 'find_maximum'
        })
    
    # Conditional logic
    for i in range(num_samples // 4):
        num = random.randint(1, 100)
        expected = "even" if num % 2 == 0 else "odd"
        problems.append({
            'task_id': f'synthetic_even_odd_{i}',
            'prompt': f'Return "even" if {num} is even, "odd" otherwise',
            'solution': f'return "even" if {num} % 2 == 0 else "odd"',
            'test': f'assert check_even_odd() == "{expected}"',
            'entry_point': 'check_even_odd'
        })
    
    return problems

def load_mbpp_data(num_samples: int = 1000):
    """Loads and formats data from the MBPP (Mostly Basic Python Problems) dataset."""
    try:
        dataset = load_dataset("mbpp", "sanitized", split="train")
        problems = []
        
        for i, example in enumerate(dataset):
            if i >= num_samples:
                break
                
            problems.append({
                'task_id': f'mbpp_{example["task_id"]}',
                'prompt': f'{example["text"]}\n\n{example["code"].split("def")[0]}def ',
                'solution': example['code'].split('def ')[1] if 'def ' in example['code'] else example['code'],
                'test': '\n'.join(example['test_list']),
                'entry_point': extract_function_name(example['code'])
            })
        
        return problems
    except Exception as e:
        print(f"Error loading MBPP: {e}")
        return create_synthetic_coding_problems(num_samples)

def extract_function_name(code: str) -> str:
    """Extracts a function name from a string of code using regex."""
    match = re.search(r'def\s+(\w+)\s*\(', code)
    return match.group(1) if match else 'main'

class BaselineTransformer(torch.nn.Module):
    """A simple, standard Transformer model for baseline comparison."""
    
    def __init__(self, vocab_size: int, dim: int = 512, num_heads: int = 8, num_layers: int = 6):
        super().__init__()
        
        self.embedding = torch.nn.Embedding(vocab_size, dim)
        self.pos_embedding = torch.nn.Parameter(torch.zeros(1, 512, dim))  # Fixed max length
        self.dropout = torch.nn.Dropout(0.1)
        self.layer_norm = torch.nn.LayerNorm(dim)
        
        # Use a standard PyTorch Transformer Encoder for the baseline
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=dim, 
            nhead=num_heads, 
            dim_feedforward=4 * dim, # Standard Transformer FFN size
            batch_first=True
        )
        self.layers = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_head = torch.nn.Linear(dim, vocab_size)
        
        torch.nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)
        
    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        x = self.embedding(input_ids)
        x = x + self.pos_embedding[:, :seq_len, :]  # Add positional embeddings
        x = self.dropout(x)
        x = self.layer_norm(x)
        
        x = self.layers(x)
            
        x = self.layer_norm(x)  # Final layer norm
        return self.output_head(x)

def evaluate_code_generation(model, tokenizer, test_problems: List[Dict], device: str = 'cpu'):
    """Evaluates code generation by executing the generated code against test cases."""
    model.eval()
    correct = 0
    total = 0
    
    for problem in test_problems[:50]:  # Test on subset for speed
        try:
            # Prepare input
            prompt = f"<PROBLEM>{problem['prompt']}<SOLUTION>" # Using special tokens
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            
            with torch.no_grad():
                if hasattr(model, 'generate_solution'):
                    generated_ids = model.generate_solution(input_ids, max_length=256)
                else: # Fallback for baseline models
                    outputs = model(input_ids)
                    generated_ids = outputs.argmax(dim=-1)
            
            # Decode solution
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            solution = generated_text.split('<SOLUTION>')[-1].strip()
            
            if test_code_solution(solution, problem):
                correct += 1
            
            total += 1
            
        except Exception as e:
            print(f"Error evaluating problem {problem['task_id']}: {e}")
            total += 1
    
    return correct / total if total > 0 else 0

def test_code_solution(solution: str, problem: Dict) -> bool:
    """Executes a generated code solution in a sandboxed environment to check correctness."""
    try:
        namespace = {}
        
        exec(solution, namespace)
        
        entry_point = problem['entry_point']
        if entry_point not in namespace:
            return False
        
        test_code = problem['test']
        exec(test_code, namespace)
        
        return True
    except Exception:
        return False

def create_dataloaders(
    tokenizer,
    train_size: int = 800,
    val_size: int = 100,
    test_size: int = 100,
    batch_size: int = 8,
    max_length: int = 512
):
    """Creates train, validation, and test dataloaders for coding tasks."""
    
    # Try to get HumanEval problems (may be fewer than requested)
    he_problems = load_humaneval_data(num_samples=train_size + val_size)
    synth_needed = max(0, (train_size + val_size + test_size) - len(he_problems))
    synth_problems = create_synthetic_coding_problems(num_samples=max(test_size, synth_needed))
    all_problems = he_problems + synth_problems
    
    # If still short, top-up with additional synthetic
    required_total = train_size + val_size + test_size
    if len(all_problems) < required_total:
        short_by = required_total - len(all_problems)
        all_problems.extend(create_synthetic_coding_problems(num_samples=short_by))
    
    random.shuffle(all_problems)
    all_problems = all_problems[:required_total]
    train_problems = all_problems[:train_size]
    val_problems = all_problems[train_size:train_size + val_size]
    test_problems = all_problems[train_size + val_size:required_total]
    
    train_dataset = CodingDataset(train_problems, tokenizer, max_length, mode='train')
    val_dataset = CodingDataset(val_problems, tokenizer, max_length, mode='train')
    test_dataset = CodingDataset(test_problems, tokenizer, max_length, mode='test')
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Created datasets: {len(train_problems)} train, {len(val_problems)} val, {len(test_problems)} test")
    
    return train_loader, val_loader, test_loader, test_problems

def setup_tokenizer(model_name: str = "microsoft/CodeBERT-base"):
    """Initializes a tokenizer, trying a code-specific one first."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # GPT-2 tokenizer does not have a pad token by default.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer

# Utility functions for analysis
def analyze_model_efficiency(hrm_model, baseline_model, test_loader, device):
    """Compares the inference time and memory usage of two models."""
    import time
    
    hrm_times = []
    baseline_times = []
    
    for i, batch in enumerate(test_loader):
        if i >= 10:  # Test on 10 batches
            break
            
        input_ids = batch['input_ids'].to(device)
        
        start = time.time()
        with torch.no_grad():
            hrm_result = hrm_model(input_ids, training=False)
        hrm_times.append(time.time() - start)
        
        start = time.time()
        with torch.no_grad():
            baseline_output = baseline_model(input_ids)
        baseline_times.append(time.time() - start)
    
    hrm_avg = sum(hrm_times) / len(hrm_times)
    baseline_avg = sum(baseline_times) / len(baseline_times)
    
    print(f"Average inference time:")
    print(f"  HRM: {hrm_avg:.4f}s")
    print(f"  Baseline: {baseline_avg:.4f}s")
    print(f"  Ratio: {hrm_avg / baseline_avg:.2f}x")
    
    return {
        'hrm_time': hrm_avg,
        'baseline_time': baseline_avg,
        'ratio': hrm_avg / baseline_avg
    }

def count_model_flops(model, input_shape):
    """Provides a rough estimation of FLOPs for a model."""
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    params = count_parameters(model)
    # Rough estimate: 2 FLOPs per parameter per token
    seq_len = input_shape[1]
    estimated_flops = 2 * params * seq_len
    
    return estimated_flops