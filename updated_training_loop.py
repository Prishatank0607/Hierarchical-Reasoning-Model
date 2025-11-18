import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from hrm_core import HierarchicalReasoningModel, count_parameters
from coding_dataset import setup_tokenizer, create_dataloaders, BaselineTransformer
from hrm_training import HRMTrainer
from evaluate_coding import run_comprehensive_evaluation
import os
from datetime import datetime

def train_code_generation_models():
    """
    A standalone, complete training pipeline for comparing HRM vs. Baseline.
    Note: This is an alternative to `quick_start_guide.py` and is not used by it.
    """
    
    print("Starting HRM Code Generation Training")
    print("=" * 50)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = setup_tokenizer()
    vocab_size = len(tokenizer)
    print(f"Vocabulary size: {vocab_size}")
    train_loader, val_loader, test_loader, test_problems = create_dataloaders(
        tokenizer,
        train_size=800,    # Training problems (see problem description only)
        val_size=100,      # Validation problems  
        test_size=100,     # Test problems (for final evaluation)
        batch_size=4,      # Small batch size for code generation
        max_length=512     # Longer sequences for code
    )
    
    print(f"Data loaded: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    print("\nInitializing models...")
    
    hrm_model = HierarchicalReasoningModel(
        vocab_size=vocab_size,
        dim=256,           # Reasonable size for experiments
        num_heads=8,
        num_layers=2,
        H_cycles=3,        # N in paper 
        L_cycles=2,        # T in paper
        halt_max_steps=4
    ).to(device)
    
    baseline_model = BaselineTransformer(
        vocab_size=vocab_size,
        dim=256,
        num_heads=8,
        num_layers=4       # Comparable parameter count
    ).to(device)
    
    hrm_params = count_parameters(hrm_model)
    baseline_params = count_parameters(baseline_model)
    
    print(f"HRM parameters: {hrm_params:,}")
    print(f"Baseline parameters: {baseline_params:,}")
    print(f"Parameter ratio: {hrm_params/baseline_params:.2f}")
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"./code_generation_exp_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Experiment directory: {exp_dir}")
    
    # Train HRM with proper code generation loss
    print("\nTraining HRM...")
    hrm_model.tokenizer = tokenizer
    hrm_optimizer = torch.optim.AdamW(hrm_model.parameters(), lr=1e-4, weight_decay=0.01)
    hrm_scheduler = torch.optim.lr_scheduler.LinearLR(hrm_optimizer, start_factor=0.1, total_iters=100)
    
    best_hrm_loss = float('inf')
    
    for epoch in range(10):  # Train for 10 epochs
        hrm_model.train()
        total_loss = 0
        
        print(f"\nEpoch {epoch+1}/10")
        
        for i, batch in enumerate(train_loader):
            if i >= 100:  # Limit training batches for speed
                break
                
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)
            
            hrm_optimizer.zero_grad()
            
            result = hrm_model(input_ids, max_segments=3, training=True)
            final_output = result['outputs'][-1]
            
            # Causal LM loss: predict the next token.
            # Shift targets to align with predictions
            shift_logits = final_output[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
            
            if len(result['q_values']) > 1:
                # A simple penalty for continuing, to encourage efficient halting.
                q_loss = sum(torch.mean(q_vals[:, 1]) for q_vals in result['q_values'][:-1])  # Continue penalty
                loss = loss + 0.1 * q_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(hrm_model.parameters(), 1.0)
            hrm_optimizer.step()
            
            total_loss += loss.item()
            
            if i % 20 == 0:
                avg_segments = sum(result['segments_used'] for result in [result]) / 1
                print(f"  Batch {i:3d}, Loss: {loss.item():.4f}, Segments: {avg_segments:.1f}")
        
        hrm_scheduler.step()
        
        hrm_model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                targets = batch['targets'].to(device)
                
                result = hrm_model(input_ids, max_segments=3, training=False)
                final_output = result['outputs'][-1]
                
                shift_logits = final_output[..., :-1, :].contiguous()
                shift_labels = targets[..., 1:].contiguous()
                
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
                
                val_loss += loss.item()
                val_batches += 1
                
                if val_batches >= 20:  # Limit validation for speed
                    break
        
        avg_train_loss = total_loss / min(100, len(train_loader))
        avg_val_loss = val_loss / val_batches
        
        print(f"  Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_hrm_loss:
            best_hrm_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': hrm_model.state_dict(),
                'optimizer_state_dict': hrm_optimizer.state_dict(),
                'val_loss': avg_val_loss
            }, f"{exp_dir}/best_hrm_model.pt")
            print(f"  Saved best HRM model (val_loss: {avg_val_loss:.4f})")
    
    # Train baseline model
    print("\nTraining Baseline...")
    baseline_optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=1e-4, weight_decay=0.01)
    best_baseline_loss = float('inf')
    
    for epoch in range(8):  # Train baseline for fewer epochs
        baseline_model.train()
        total_loss = 0
        
        print(f"\nBaseline Epoch {epoch+1}/8")
        
        for i, batch in enumerate(train_loader):
            if i >= 100:
                break
                
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)
            
            baseline_optimizer.zero_grad()
            
            output = baseline_model(input_ids)
            
            # Use the same causal LM loss as the HRM.
            shift_logits = output[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(baseline_model.parameters(), 1.0)
            baseline_optimizer.step()
            
            total_loss += loss.item()
            
            if i % 20 == 0:
                print(f"  Batch {i:3d}, Loss: {loss.item():.4f}")
        
        avg_train_loss = total_loss / min(100, len(train_loader))
        print(f"  Train Loss: {avg_train_loss:.4f}")
        
        if avg_train_loss < best_baseline_loss:
            best_baseline_loss = avg_train_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': baseline_model.state_dict(),
                'optimizer_state_dict': baseline_optimizer.state_dict(),
                'train_loss': avg_train_loss
            }, f"{exp_dir}/best_baseline_model.pt")
    
    print("\nTraining completed!")
    print(f"Models saved in: {exp_dir}")
    
    print("\nRunning comprehensive evaluation...")
    results = run_comprehensive_evaluation(
        model_path=f"{exp_dir}/best_hrm_model.pt",
        baseline_path=f"{exp_dir}/best_baseline_model.pt"
    )
    
    return results, exp_dir

def quick_test_generation_fixed():
    """A quick test to check text generation with different temperature settings."""
    print("Running improved generation test...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeBERT-base")
        print("Using CodeBERT tokenizer")
    except:
        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            print("Using GPT-2 tokenizer")
        except:
            print("Error: Cannot load tokenizer")
            return
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = HierarchicalReasoningModel(
        vocab_size=len(tokenizer),
        dim=128,
        num_heads=4,
        num_layers=1,
        H_cycles=2,
        L_cycles=2
    ).to(device)
    
    model.tokenizer = tokenizer
    
    with torch.no_grad():
        nn.init.normal_(model.embed_tokens.weight, std=0.01)
        nn.init.normal_(model.lm_head.weight, std=0.01)
    
    # Test with multiple prompts.
    test_cases = [
        {
            'prompt': "def add_numbers():\n    \"\"\"Return 5 + 3\"\"\"\n    ",
            'expected': "return 5 + 3"
        },
        {
            'prompt': "def get_length():\n    \"\"\"Return length of 'hello'\"\"\"\n    ",
            'expected': "return len('hello')"
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest {i+1}:")
        print(f"Input: {repr(test_case['prompt'])}")
        
        input_ids = tokenizer.encode(
            test_case['prompt'], 
            return_tensors='pt',
            add_special_tokens=False
        ).to(device)
        
        model.eval()
        strategies = [
            {'temp': 0.1, 'name': 'Low temp (0.1)'},
            {'temp': 0.5, 'name': 'Medium temp (0.5)'},
            {'temp': 1.0, 'name': 'High temp (1.0)'}
        ]
        
        for strategy in strategies:
            try:
                with torch.no_grad():
                    generated = model.generate(
                        input_ids, 
                        max_length=input_ids.shape[1] + 20,  # Only generate 20 more tokens
                        temperature=strategy['temp']
                    )
                    
                    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
                    solution = generated_text[len(test_case['prompt']):].strip()
                    
                    print(f"  {strategy['name']}: {repr(solution[:50])}")  # First 50 chars
                    
            except Exception as e:
                print(f"  {strategy['name']}: Error - {e}")

def test_tokenizer_behavior():
    """A utility to see how different tokenizers handle snippets of code."""
    print("Testing tokenizer behavior on code...")
    
    code_samples = [
        "return 5 + 3",
        "def function():",
        "    pass",
        "import numpy as np"
    ]
    
    tokenizers_to_test = [
        ("gpt2", "GPT-2"),
        ("microsoft/CodeBERT-base", "CodeBERT"),
    ]
    
    for model_name, display_name in tokenizers_to_test:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            print(f"\n{display_name} tokenization:")
            
            for code in code_samples:
                tokens = tokenizer.encode(code, add_special_tokens=False)
                decoded = tokenizer.decode(tokens)
                print(f"  '{code}' -> {len(tokens)} tokens -> '{decoded}'")
                
        except Exception as e:
            print(f"  {display_name}: Failed to load - {e}")

def minimal_training_test():
    """Trains a tiny model for a few steps on a simple task to verify the training loop."""
    print("Running minimal training test...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = setup_tokenizer()
    model = HierarchicalReasoningModel(
        vocab_size=len(tokenizer),
        dim=64,
        num_heads=4,
        num_layers=1,
        H_cycles=1,
        L_cycles=1
    ).to(device)
    
    model.tokenizer = tokenizer
    
    examples = [
        ("def add():\n    \"\"\"Return 2 + 3\"\"\"\n    ", "return 2 + 3"),
        ("def sub():\n    \"\"\"Return 5 - 1\"\"\"\n    ", "return 5 - 1"),
        ("def mul():\n    \"\"\"Return 4 * 2\"\"\"\n    ", "return 4 * 2"),
    ]
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("Training for 50 steps...")
    model.train()
    
    for step in range(50):
        total_loss = 0
        
        for prompt, target in examples:
            full_text = prompt + target
            tokens = tokenizer.encode(full_text, return_tensors='pt').to(device)
            prompt_tokens = tokenizer.encode(prompt, return_tensors='pt').to(device)
            
            result = model(tokens, max_segments=2, training=True)
            logits = result['outputs'][-1]
            
            target_start = prompt_tokens.shape[1] # Calculate where the target text begins.
            target_logits = logits[:, target_start-1:-1, :]  # Shift by 1 for causal LM
            target_tokens = tokens[:, target_start:]
            
            loss = F.cross_entropy(
                target_logits.contiguous().view(-1, target_logits.size(-1)),
                target_tokens.contiguous().view(-1)
            )
            
            total_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if step % 10 == 0:
            print(f"  Step {step}: Loss = {total_loss/len(examples):.4f}")
    
    print("\nTesting generation after minimal training:")
    test_prompt = "def add():\n    \"\"\"Return 2 + 3\"\"\"\n    "
    input_ids = tokenizer.encode(test_prompt, return_tensors='pt').to(device)
    
    model.eval()
    with torch.no_grad():
        generated = model.generate(input_ids, max_length=input_ids.shape[1] + 10, temperature=0.3)
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        solution = generated_text[len(test_prompt):].strip()
        
        print(f"Generated: {repr(solution)}")

if __name__ == "__main__":
    print("HRM Generation Debugging")
    print("Choose test:")
    print("1. Fixed generation test")
    print("2. Tokenizer behavior test") 
    print("3. Minimal training test")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        quick_test_generation_fixed()
    elif choice == "2":
        test_tokenizer_behavior()
    elif choice == "3":
        minimal_training_test()
    else:
        print("Running all tests...")
        test_tokenizer_behavior()
        quick_test_generation_fixed()
        minimal_training_test()