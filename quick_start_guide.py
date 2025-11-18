# ============================================================================
# QUICK START GUIDE - Run this file to test the setup and experiments.
# ============================================================================

def quick_test():
    """Quick test to verify everything is working"""
    print("🧪 Running quick HRM test...")
    
    # Basic imports
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer
    
    # Check if our modules can be imported
    try:
        from hrm_core import HierarchicalReasoningModel
        from hrm_training import HRMTrainer
        from coding_dataset import setup_tokenizer, create_synthetic_coding_problems
        print("✅ All modules imported successfully!")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Test tokenizer
    tokenizer = setup_tokenizer()
    print(f"📚 Tokenizer loaded: {len(tokenizer)} tokens")
    
    # Test model creation
    model = HierarchicalReasoningModel(
        vocab_size=len(tokenizer),
        dim=64,      # Very small for testing
        num_heads=4,
        num_layers=1,
        H_cycles=2,  # N in paper - high-level cycles
        L_cycles=2   # T in paper - low-level cycles per H cycle
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model created: {param_count:,} parameters")
    
    # Test forward pass
    test_input = torch.randint(0, len(tokenizer), (1, 32)).to(device)
    with torch.no_grad():
        output = model(test_input, training=False)
    
    print(f"✅ Forward pass successful!")
    print(f"   Segments used: {output['segments_used']}")
    print(f"   Output shape: {output['outputs'][0].shape}")
    
    print("\n🎉 Quick test completed successfully!")
    print("💡 You can now run the full experiments!")
    return True

# ============================================================================
# ADVANCED EXPERIMENT CONFIGURATIONS
# ============================================================================

class ExperimentConfig:
    """Configuration for different experiment types"""
    
    # Tiny experiment
    TINY = {
        'model_params': {
            'dim': 64,
            'num_heads': 4,
            'num_layers': 1,
            'H_cycles': 2,
            'L_cycles': 2,
            'residual_scale': 0.9,
            'adaptive_cycles': True
        },
        'data_params': {
            'train_size': 150,
            'val_size': 20,
            'test_size': 20,
            'batch_size': 4,
            'max_length': 128
        },
        'training_params': {
            'epochs': 8,
            'learning_rate': 1.2e-3,
            'warmup_steps': 50,
            'weight_decay': 0.05,
            'lr_schedule': 'cosine',
            'min_lr': 1e-5,
            'patience': 3,
            'gradient_accumulation_steps': 2
        }
    }
    
    # Small experiment
    SMALL = {
        'model_params': {
            'dim': 128,
            'num_heads': 8,
            'num_layers': 2,
            'H_cycles': 3,  # More planning
            'L_cycles': 2,  # Same execution
            'residual_scale': 0.9,
            'adaptive_cycles': True
        },
        'data_params': {
            'train_size': 600,
            'val_size': 100,
            'test_size': 100,
            'batch_size': 16,
            'max_length': 192,
            'gradient_accumulation_steps': 2
        },
        'training_params': {
            'epochs': 20,
            'learning_rate': 3e-4,
            'warmup_steps': 100,
            'weight_decay': 0.05,
            'lr_schedule': 'cosine',
            'min_lr': 1e-5,
            'patience': 3
        }
    }
    
    # Medium experiment
    MEDIUM = {
        'model_params': {
            'dim': 192,      # Reduced for stability
            'num_heads': 8,
            'num_layers': 4, # Increased to compensate
            'H_cycles': 3,  # Reduced for stability
            'L_cycles': 2,  # Same execution
            'residual_scale': 0.85,
            'adaptive_cycles': True
        },
        'data_params': {
            'train_size': 800,
            'val_size': 150,
            'test_size': 150,
            'batch_size': 8,
            'max_length': 256,
            'gradient_accumulation_steps': 4
        },
        'training_params': {
            'epochs': 30,
            'learning_rate': 2.5e-4,
            'warmup_steps': 200,
            'weight_decay': 0.06,
            'lr_schedule': 'cosine',
            'min_lr': 1e-5,
            'gradient_clip': 1.0,
            'use_amp': True,
            'gradient_checkpointing': True
        }
    }

def run_experiment(config_name='SMALL', save_results=True):
    """Run a complete experiment with a given configuration."""
    
    import torch
    import torch.nn.functional as F
    from hrm_core import HierarchicalReasoningModel, count_parameters
    from hrm_training import HRMTrainer, compare_with_baseline
    from coding_dataset import setup_tokenizer, create_dataloaders, BaselineTransformer
    import json
    import os
    from datetime import datetime
    from tqdm import tqdm
    
    # Get configuration
    config = getattr(ExperimentConfig, config_name)
    print(f"🚀 Running {config_name} experiment")
    print(f"Config: {config}")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = setup_tokenizer()
    vocab_size = len(tokenizer)
    
    # Create data. `gradient_accumulation_steps` is a training param, not a data param.
    data_params = config['data_params'].copy()
    data_params.pop('gradient_accumulation_steps', None)  # Remove if it exists
    
    train_loader, val_loader, test_loader, test_problems = create_dataloaders(
        tokenizer, **data_params
    )
    
    # Create HRM model with all parameters
    hrm_model = HierarchicalReasoningModel(
        vocab_size=vocab_size,
        **config['model_params'],
        gradient_checkpointing=config['training_params'].get('gradient_checkpointing', False)
    ).to(device)
    hrm_params = count_parameters(hrm_model) # Used for parameter-matching the baseline

    # ====== ROBUST PARAMETER-MATCHED BASELINE GENERATION ======
    import math
    def calculate_baseline_config(hrm_params, hrm_config, vocab_size):
        """
        Dynamically calculates a baseline model configuration to closely match the
        parameter count of the HRM model. It iterates through layer counts and
        calculates the ideal dimension for each, selecting the best match.
        """
        best_config = None
        min_diff_percent = float('inf')
        
        # Define search space for layers
        for num_layers in range(2, 16):
            # The total parameters can be modeled as a quadratic equation in `dim`:
            # a*dim^2 + b*dim - c = 0. We solve for `dim` to match the target `hrm_params`.
            
            # Coefficients for the quadratic formula
            # Standard Transformer: 4d^2 (MHA) + 8d^2 (FFN) = 12d^2 per layer
            a = 12 * num_layers
            b = 2 * vocab_size + 512  # From input/output embeddings + learned pos_embedding
            c = -hrm_params
            
            # Solve for dim using the quadratic formula: (-b + sqrt(b^2 - 4ac)) / 2a
            discriminant = b**2 - 4*a*c
            if discriminant < 0:
                continue # No real solution for this layer count
            
            # Calculate ideal dimension and round to a multiple of 16 for efficiency
            ideal_dim = (-b + math.sqrt(discriminant)) / (2 * a)
            test_dim = round(ideal_dim / 16) * 16
            if test_dim == 0: continue

            # Determine a reasonable number of heads
            head_dim_target = 64 # A common head dimension
            test_heads = max(2, round(test_dim / head_dim_target))
            # Ensure `dim` is divisible by `heads` and that `heads` is even for optimization
            while test_heads > 1:
                if test_dim % test_heads == 0 and test_heads % 2 == 0:
                    break # Found a valid number of heads
                test_heads -= 1
            
            # Create a temporary model to get the exact parameter count
            tmp_model = BaselineTransformer(
                vocab_size=vocab_size,
                dim=test_dim,
                num_heads=test_heads,
                num_layers=num_layers
            )
            test_params = count_parameters(tmp_model)
            del tmp_model

            diff_percent = abs(test_params - hrm_params) / hrm_params * 100
            if diff_percent < min_diff_percent:
                min_diff_percent = diff_percent
                best_config = (test_dim, test_heads, num_layers, test_params)
        
        return best_config

    base_dim = config['model_params']['dim']
    base_heads = config['model_params']['num_heads']
    base_layers = config['model_params']['num_layers']
    matched_baseline_cfg = calculate_baseline_config(hrm_params, config, vocab_size)
    if matched_baseline_cfg:
        baseline_dim, baseline_heads, baseline_layers, baseline_params = matched_baseline_cfg
        print(f"📊 Matched Baseline: dim={baseline_dim}, heads={baseline_heads}, layers={baseline_layers}, params={baseline_params:,}")
        baseline_model = BaselineTransformer(
            vocab_size=vocab_size,
            dim=baseline_dim,
            num_heads=baseline_heads,
            num_layers=baseline_layers
        ).to(device)
    else:
        print("[Warning] Could not parameter-match baseline within search space. Using config defaults.")
        baseline_model = BaselineTransformer(
            vocab_size=vocab_size,
            dim=base_dim,
            num_heads=base_heads,
            num_layers=base_layers
        ).to(device)
        baseline_params = count_parameters(baseline_model)

    print(f"📊 HRM Parameters: {hrm_params:,}")
    print(f"📊 Baseline Parameters: {baseline_params:,}")
    print(f"🧠 H-cycles (Planner): {config['model_params']['H_cycles']}")
    print(f"⚡ L-cycles (Executor): {config['model_params']['L_cycles']}")
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"./experiments/{config_name}_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create HRM trainer
    trainer_params = {
        'learning_rate': config['training_params']['learning_rate'],
        'weight_decay': config['training_params'].get('weight_decay', 0.05),
        'max_segments': 2 if config_name == 'TINY' else 4,
        'min_segments': 1,
        'memory_efficient': True,
        'gradient_approximation_steps': 2,
        'use_amp': config['training_params'].get('use_amp', False)
    }
        
    hrm_trainer = HRMTrainer(hrm_model, device=device, **trainer_params)
    
    # Train baseline with improved training process
    print(f"\n🎯 Training Baseline Model...")
    
    # Get training parameters
    baseline_epochs = config['training_params']['epochs']
    warmup_steps = config['training_params'].get('warmup_steps', 100)
    num_training_steps = baseline_epochs * len(train_loader)
    
    # Setup optimizers and schedulers
    baseline_optimizer = torch.optim.AdamW(
        baseline_model.parameters(),
        lr=config['training_params']['learning_rate'],
        weight_decay=config['training_params'].get('weight_decay', 0.05),
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))  # Cosine decay
        
    baseline_scheduler = torch.optim.lr_scheduler.LambdaLR(baseline_optimizer, lr_lambda)
    
    # Baseline training loop
    best_val_acc = 0
    # Use the scaler from the HRM trainer for convenience
    grad_accum_steps = config['training_params'].get('gradient_accumulation_steps', 1)
    
    for epoch in range(baseline_epochs):
        baseline_model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Baseline Epoch {epoch+1}")):
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)
            
            with torch.amp.autocast(device_type='cuda', enabled=hrm_trainer.use_amp):
                output = baseline_model(input_ids)
                shift_logits = output[..., :-1, :].contiguous()
                shift_labels = targets[..., 1:].contiguous()
                
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
                
                # Scale loss for gradient accumulation
                scaled_loss = loss / grad_accum_steps
            
            hrm_trainer.scaler.scale(scaled_loss).backward()
            
            if (batch_idx + 1) % grad_accum_steps == 0 or batch_idx == len(train_loader) - 1:
                hrm_trainer.scaler.unscale_(baseline_optimizer)
                torch.nn.utils.clip_grad_norm_(baseline_model.parameters(), 1.0)
                hrm_trainer.scaler.step(baseline_optimizer)
                hrm_trainer.scaler.update()
                baseline_optimizer.zero_grad()
                baseline_scheduler.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"  Baseline Epoch {epoch+1}: {avg_loss:.4f}")
        
        # Baseline validation
        baseline_model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                targets = batch['targets'].to(device)
                
                with torch.amp.autocast(device_type='cuda', enabled=hrm_trainer.use_amp):
                    output = baseline_model(input_ids)
                    shift_logits = output[..., :-1, :].contiguous()
                    shift_labels = targets[..., 1:].contiguous()
                    
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100
                    )
                    val_loss += loss.item()
                    
                    # Calculate accuracy
                    pred = shift_logits.argmax(dim=-1)
                    mask = (shift_labels != -100)
                    val_correct += (pred[mask] == shift_labels[mask]).sum().item()
                    val_total += mask.sum().item()
        
        val_acc = val_correct / val_total if val_total > 0 else 0
        print(f"  Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(baseline_model.state_dict(), f"{exp_dir}/best_baseline.pt")
            torch.save({ # Save full checkpoint for potential resumption
                'model_state_dict': baseline_model.state_dict(),
                'optimizer_state_dict': baseline_optimizer.state_dict(),
                'scheduler_state_dict': baseline_scheduler.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc
            }, f"{exp_dir}/baseline_checkpoint.pt")
            print(f"  💾 New best baseline: {val_acc:.4f}")
    
    # Load best model for final evaluation
    best_baseline_state = torch.load(f"{exp_dir}/best_baseline.pt")
    baseline_model.load_state_dict(best_baseline_state)
    
    # Train HRM
    print(f"\n🧠 Training HRM Model...")
    
    best_val_acc = 0
    training_log = []
    epochs = config['training_params']['epochs']
    
    for epoch in range(epochs):
        hrm_trainer.current_epoch = epoch
        
        hrm_trainer.update_residual_scaling(epochs) # Progressive residual scaling
        
        print(f"\nEpoch {epoch+1}/{epochs}")

        hrm_model.train()
        epoch_metrics = {'task_loss': 0, 'q_loss': 0, 'accuracy': 0, 'segments_used': 0}
        
        for batch in tqdm(train_loader, desc="Training"):
            metrics = hrm_trainer.train_step(batch, total_epochs=epochs)
            for key in epoch_metrics:
                if key in metrics:
                    epoch_metrics[key] += metrics[key]
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= len(train_loader)
        
        # HRM validation
        val_metrics = hrm_trainer.evaluate(val_loader)
        hrm_trainer.scheduler.step(val_metrics['loss'])
        
        # Log results
        log_entry = {
            'epoch': epoch + 1,
            'train_loss': epoch_metrics['task_loss'],
            'train_acc': epoch_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
            'avg_segments': val_metrics['avg_segments'],
        }
        training_log.append(log_entry)
        
        print(f"  Train Loss: {epoch_metrics['task_loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Avg Segs: {val_metrics['avg_segments']:.1f}")
        
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save(hrm_model.state_dict(), f"{exp_dir}/best_hrm.pt")
            torch.save({ # Save full checkpoint for potential resumption
                'model_state_dict': hrm_model.state_dict(),
                'optimizer_state_dict': hrm_trainer.optimizer.state_dict(),
                'epoch': epoch
            }, f"{exp_dir}/hrm_checkpoint.pt")
            print(f"  💾 New best: {best_val_acc:.4f}")
    
    # Final evaluation
    print(f"\n📊 Final Evaluation...")
    best_hrm_state = torch.load(f"{exp_dir}/best_hrm.pt")
    hrm_model.load_state_dict(best_hrm_state)
    baseline_model.load_state_dict(torch.load(f"{exp_dir}/best_baseline.pt"))
    
    # Evaluate both models on the validation set
    def evaluate_model(model, is_hrm=False):
        model.eval()
        correct = 0
        total = 0
        total_segments = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                targets = batch['targets'].to(device)
                
                if is_hrm:
                    result = model(input_ids, max_segments=hrm_trainer.max_segments, training=False)
                    output_logits = result['outputs'][-1]
                    total_segments += result['segments_used']
                    output = output_logits[..., :-1, :].contiguous()
                    targets = targets[..., 1:].contiguous()
                else:
                    # Align logits and labels for correct accuracy calculation
                    output_logits = model(input_ids)
                    output = output_logits[..., :-1, :].contiguous()
                    targets = targets[..., 1:].contiguous()
                
                pred = output.argmax(dim=-1)
                
                mask = targets != -100
                masked_total = mask.sum().item()
                if masked_total > 0:
                    correct += (pred[mask] == targets[mask]).sum().item() 
                total += masked_total
        
        accuracy = correct / total if total > 0 else 0
        avg_segments = total_segments / len(val_loader) if is_hrm else 0
        return accuracy, avg_segments
    
    hrm_accuracy, avg_segments = evaluate_model(hrm_model, is_hrm=True)
    baseline_accuracy, _ = evaluate_model(baseline_model, is_hrm=False)
    
    # Results
    if baseline_accuracy == 0:
        relative_improvement = None
    else:
        relative_improvement = (hrm_accuracy - baseline_accuracy) / baseline_accuracy * 100

    results = {
        'config': config,
        'hrm_params': hrm_params,
        'baseline_params': baseline_params,
        'hrm_accuracy': hrm_accuracy,
        'baseline_accuracy': baseline_accuracy,
        'improvement': hrm_accuracy - baseline_accuracy,
        'relative_improvement': relative_improvement,
        'avg_segments': avg_segments,
        'timestamp': timestamp
    }
    
    print(f"\n🏆 RESULTS:")
    print(f"HRM Accuracy: {hrm_accuracy:.4f}")
    print(f"Baseline Accuracy: {baseline_accuracy:.4f}")
    print(f"Improvement: {results['improvement']:+.4f}")
    if results['relative_improvement'] is not None:
        print(f"Relative Improvement: {results['relative_improvement']:+.1f}%")
    else:
        print("Relative Improvement: N/A (baseline accuracy is zero)")
    
    if hrm_accuracy > baseline_accuracy:
        print("\n🎉 SUCCESS! HRM outperforms baseline!")
    else: # Note: This can happen in early/short training runs.
        print(f"\n⚠️  HRM still behind baseline")
    
    # Save results
    if save_results:
        with open(f"{exp_dir}/results.json", 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to ./experiments/{config_name}_{timestamp}/results.json")
    
    return results

def ablation_study_N_T():
    """Study the effect of different H_cycles (N) and L_cycles (T) values."""
    print("🔬 Running N/T ablation study...")
    
    configurations = [
        {'N': 2, 'T': 2},
        {'N': 3, 'T': 2},
        {'N': 4, 'T': 2},
        {'N': 2, 'T': 3},
        {'N': 3, 'T': 3},
    ]
    
    results = []
    base_config = ExperimentConfig.TINY  # Use tiny config for speed
    
    for params in configurations:
        print(f"\n🧪 Testing N={params['N']}, T={params['T']}")
        
        # Modify config
        config = base_config.copy()
        config['model_params'] = base_config['model_params'].copy()
        config['data_params'] = base_config['data_params'].copy()
        config['training_params'] = base_config['training_params'].copy()
        config['model_params']['H_cycles'] = params['N']
        config['model_params']['L_cycles'] = params['T']
        
        try:
            result = run_experiment_quick(config)
            result['N'] = params['N']
            result['T'] = params['T']
            results.append(result)
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    # Analyze results
    print("\n📊 Ablation Results:")
    print("N  T  | Params  | Accuracy | Improvement")
    print("-" * 40)
    for r in results:
        print(f"{r['N']}  {r['T']}  | {r['hrm_params']//1000:4d}K | {r['hrm_accuracy']:.4f}   | {r['improvement']:+.4f}")
    
    return results

def run_experiment_quick(config):
    """A simplified, faster experiment run for ablation studies."""
    # Simplified version of run_experiment for speed
    import torch
    from hrm_core import HierarchicalReasoningModel, count_parameters
    from coding_dataset import setup_tokenizer, create_dataloaders
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = setup_tokenizer()
    
    print("Loading coding problems...")
    train_loader, val_loader, test_loader, test_problems = create_dataloaders(
        tokenizer,
        train_size=config['data_params']['train_size'],
        val_size=config['data_params']['val_size'],
        test_size=config['data_params']['test_size'],
        batch_size=config['data_params']['batch_size'],
        max_length=config['data_params']['max_length']
    )
    
    # Quick training on a few batches
    model = HierarchicalReasoningModel(
        vocab_size=len(tokenizer),
        dim=config['model_params']['dim'],
        num_heads=config['model_params']['num_heads'],
        num_layers=config['model_params']['num_layers'],
        H_cycles=config['model_params']['H_cycles'],
        L_cycles=config['model_params']['L_cycles']
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    for i, batch in enumerate(train_loader):
        if i >= 10:
            break
        
        input_ids = batch['input_ids'].to(device)
        targets = batch['targets'].to(device)
        
        optimizer.zero_grad()
        result = model(input_ids, max_segments=2, training=True)
        output = result['outputs'][-1]
        
        # Correctly shift for causal LM loss
        shift_logits = output[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )
        loss.backward()
        optimizer.step()
    
    # Quick evaluation on validation set
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)
            
            result = model(input_ids, max_segments=2, training=False)
            output = result['outputs'][-1]
            
            # Correctly shift for causal LM accuracy
            shift_logits = output[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            pred = shift_logits.argmax(dim=-1)
            mask = shift_labels != -100
            correct += (pred[mask] == shift_labels[mask]).sum().item()
            total += mask.sum().item()
    
    accuracy = correct / total if total > 0 else 0
    
    return {
        'hrm_params': count_parameters(model),
        'hrm_accuracy': accuracy,
        'improvement': accuracy - 0.25  # Assume baseline of 0.25 for quick test
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("🧪 HRM Experiments - Advanced Configuration")
    print("=" * 50)
    
    # Run quick test first
    if quick_test():
        print("\n🎯 Choose your experiment:")
        print("1. TINY - Quick test (~16M params)")
        print("2. SMALL - Main experiment (~35M params)")
        print("3. MEDIUM - Larger experiment (~63M params)")
        print("4. ABLATION - N/T parameter study")

        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            run_experiment('TINY')
        elif choice == "2":
            run_experiment('SMALL')
        elif choice == "3":
            run_experiment('MEDIUM')
        elif choice == "4":
            ablation_study_N_T()
        else:
            print("Invalid choice. Please run the script again.")
    
# ============================================================================
# RESEARCH DIRECTIONS TO EXPLORE
# ============================================================================

"""
🔬 RESEARCH IDEAS TO TRY:

1. **Architecture Variations:**
   - Different H/L cycle ratios (e.g., more planning vs. execution).
   - Cross-attention between the H and L modules for richer information flow.

2. **Training Improvements:**
   - Curriculum learning (e.g., from easy to hard problems).
   - More advanced Adaptive Computation Time (ACT) exploration strategies.

3. **Evaluation Extensions:**
   - Test on longer, more complex coding problems.
   - Evaluate on different domains, like mathematical reasoning.

4. **Analysis Studies:**
   - Visualize attention patterns to understand what the model focuses on.
   - Analyze the internal representations of the H and L modules.

The key hypothesis is that hierarchical reasoning with different timescales
can outperform standard transformers of a similar size on complex tasks.
"""