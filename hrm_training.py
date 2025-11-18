import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os

class HRMTrainer:
    """A dedicated trainer for the Hierarchical Reasoning Model."""
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        max_segments: int = 4,
        min_segments: int = 1,
        epsilon: float = 0.1,  # For ACT exploration
        memory_efficient: bool = True,
        gradient_approximation_steps: int = 1,
        use_amp: bool = False
    ):
        self.model = model.to(device)
        self.device = device
        self.max_segments = max_segments
        self.min_segments = min_segments
        self.epsilon = epsilon # Probability of exploring a random number of segments.
        
        # Approximates gradients by only backpropagating through the last few segments.
        self.memory_efficient = memory_efficient
        
        # AMP setup
        self.use_amp = use_amp
        self.scaler = torch.amp.GradScaler(device='cuda', enabled=self.use_amp)
        self.gradient_approximation_steps = min(2, gradient_approximation_steps)
        
        # Use different learning rates for different parts of the HRM architecture.
        param_groups = [
            # Embeddings are trained with a lower LR for stability.
            {'params': model.embed_tokens.parameters(), 'lr': learning_rate * 0.4, 'name': 'embeddings'},
            {'params': model.H_module.parameters(), 'lr': learning_rate * 1.1, 'name': 'H_module'},
            {'params': model.L_module.parameters(), 'lr': learning_rate * 1.3, 'name': 'L_module'},
            {'params': model.lm_head.parameters(), 'lr': learning_rate * 0.9, 'name': 'lm_head'},
            {'params': model.q_head.parameters(), 'lr': learning_rate * 0.6, 'name': 'q_head'},
        ]
        
        # Add new gating layers to the optimizer
        if hasattr(model, 'h_gate'):
            param_groups.append({'params': model.h_gate.parameters(), 'lr': learning_rate, 'name': 'h_gate'})
            param_groups.append({'params': model.l_gate.parameters(), 'lr': learning_rate, 'name': 'l_gate'})
            param_groups.append({'params': model.h_feedback_gate.parameters(), 'lr': learning_rate, 'name': 'h_feedback_gate'})
        
        # Add auxiliary heads if they exist
        if hasattr(model, 'aux_lm_heads') and len(model.aux_lm_heads) > 0:
            param_groups.append({
                'params': model.aux_lm_heads.parameters(), 
                'lr': learning_rate * 0.8, 
                'name': 'aux_heads'
            })
        
        self.optimizer = optim.AdamW(
            param_groups,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            'min',      # Reduce LR when validation loss plateaus.
            patience=2, 
            factor=0.5 # Reduce LR by half
        )
        
        self.training_history = {
            'losses': [],
            'q_losses': [],
        }
        
        # Progressively increase the number of segments during training.
        self.curriculum_schedule = {
            'epochs_1_seg': 0.4,    # 40% of epochs with 1 segment (more stability)
            'epochs_2_seg': 0.4,    # 40% of epochs with 2 segments  
            'epochs_3_seg': 0.2     # 20% of epochs with 3 segments (less for tiny tasks)
        }
        
        # TINY EXPERIMENT MODE: Automatically adjust for small models
        if hasattr(self.model, 'dim') and self.model.dim <= 128:
            self.curriculum_schedule = {
                'epochs_1_seg': 0.5,    # More time with 1 segment
                'epochs_2_seg': 0.5,    # Equal time with 2 segments
                'epochs_3_seg': 0.0     # Skip 3 segments for tiny models
            }
            self.max_segments = min(2, max_segments)  # Limit segments for tiny models
        
        # A small penalty to encourage the model to halt sooner if possible.
        self.compute_penalty_weight = 1e-5
        
        # Q-loss scheduling parameters
        self.current_epoch = 0
        self.q_loss_start_epoch = 3
        self.q_loss_ramp_epochs = 15
        self.q_loss_max_weight = 0.1
        
        # Gradually increase the residual scaling factor during training.
        self.initial_residual_scale = 0.8
        self.final_residual_scale = 0.98
        
    def compute_q_targets(self, outputs: List[torch.Tensor], targets: torch.Tensor) -> List[torch.Tensor]:
        """Computes targets for the halting head using a 1-step lookahead."""
        q_targets = []
        
        for i, output in enumerate(outputs):
            batch_size = output.shape[0]
            
            # Correctly shift for causal LM evaluation
            shift_logits = output[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            
            # Compute accuracy for this segment
            predictions = shift_logits.argmax(dim=-1)
            mask = (shift_labels != -100)
            correct = (predictions[mask] == shift_labels[mask]).sum() / mask.sum()
            
            halt_reward = correct  # The immediate reward for halting is the current accuracy.
            
            if i < len(outputs) - 1:  # Not the last segment
                with torch.no_grad(): # Ensure this calculation doesn't track gradients
                    next_shift_logits = outputs[i+1][..., :-1, :].contiguous()
                    next_predictions = next_shift_logits.argmax(dim=-1)
                    next_correct = (next_predictions[mask] == shift_labels[mask]).sum() / mask.sum()
                    continue_value = torch.max(next_correct, halt_reward) # Use the max of next or current.
            else:  # Last segment - must halt
                continue_value = torch.zeros_like(halt_reward)
            
            # Create a [2] tensor for [halt_value, continue_value]
            q_targets.append(torch.stack([halt_reward, continue_value]))
        
        return q_targets
    
    def get_q_loss_weight(self) -> float:
        """Get the weight for the Q-loss, which ramps up during training."""
        if self.current_epoch < self.q_loss_start_epoch:
            return 0.0
        elif self.current_epoch < self.q_loss_start_epoch + self.q_loss_ramp_epochs:
            # Linear ramp up
            progress = (self.current_epoch - self.q_loss_start_epoch) / self.q_loss_ramp_epochs
            return progress * self.q_loss_max_weight
        else:
            return self.q_loss_max_weight
    
    def update_residual_scaling(self, total_epochs: int):
        """Gradually increase the residual scaling factor during training."""
        progress = self.current_epoch / total_epochs
        current_scale = self.initial_residual_scale + progress * (self.final_residual_scale - self.initial_residual_scale)
        
        for module in self.model.modules():
            if hasattr(module, 'residual_scale'):
                module.residual_scale = current_scale
    
    def get_adaptive_segment_weights(self, num_segments: int) -> List[float]:
        """Get segment weights that exponentially favor later, more refined, reasoning steps."""
        if num_segments == 1:
            return [1.0]
        
        weights = []
        for i in range(num_segments):
            weight = (1.5 ** i)
            weights.append(weight)
        
        total = sum(weights)
        weights = [w / total for w in weights]
        
        return weights
    
    def train_step(self, batch: Dict[str, torch.Tensor], total_epochs: int = 10) -> Dict[str, float]:
        """Single training step with deep supervision and curriculum learning"""
        self.model.train()
        self.optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(self.device)
        targets = batch['targets'].to(self.device)
        
        # Determine number of segments via curriculum or exploration.
        if np.random.random() < self.epsilon:
            num_segments = np.random.randint(self.min_segments, self.max_segments + 1)
        else:
            progress = self.current_epoch / total_epochs
            if progress < 0.4:
                num_segments = 1
            elif progress < 0.8:
                num_segments = 2
            else:
                num_segments = min(3, self.max_segments)
        
        with torch.amp.autocast(device_type='cuda', enabled=self.use_amp):
            result = self.model(input_ids, max_segments=num_segments, training=True)
            outputs = result['outputs']
            q_values_list = result['q_values']
            
            segment_weights = self.get_adaptive_segment_weights(num_segments)
            
            total_task_loss = 0
            total_q_loss = 0
            segment_losses = []
            
            # Compute loss for each segment with deep supervision
            for i, (output, q_values) in enumerate(zip(outputs, q_values_list)):
                # Correctly shift for causal LM loss
                shift_logits = output[..., :-1, :].contiguous()
                shift_labels = targets[..., 1:].contiguous()
                
                task_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
                
                # Use performance-based Q-targets for smarter halting.
                with torch.no_grad():
                    q_target_values = self.compute_q_targets(outputs, targets)
                q_loss = F.mse_loss(q_values, q_target_values[i].expand_as(q_values))
                
                # Combine task and Q-loss, weighted by segment and schedule.
                q_loss_weight = self.get_q_loss_weight()
                weighted_task_loss = segment_weights[i] * task_loss
                weighted_q_loss = segment_weights[i] * q_loss * q_loss_weight
                
                segment_losses.append(weighted_task_loss + weighted_q_loss)
                total_task_loss += weighted_task_loss.item()
                total_q_loss += weighted_q_loss.item()
            
            total_loss = sum(segment_losses)
            
            compute_penalty = self.compute_penalty_weight * num_segments
            total_loss += compute_penalty
        
        # Scale loss and backpropagate
        self.scaler.scale(total_loss).backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Report masked accuracy from the last segment for logging
        with torch.no_grad():
            final_logits = outputs[-1][..., :-1, :].contiguous()
            final_labels = targets[..., 1:].contiguous()
            
            final_pred = final_logits.argmax(dim=-1)
            final_mask = (final_labels != -100)
            masked_correct = (final_pred[final_mask] == final_labels[final_mask]).sum().item()
            masked_total = final_mask.sum().item()
            batch_acc = (masked_correct / masked_total) if masked_total > 0 else 0.0

        return {
            'task_loss': total_task_loss,
            'q_loss': total_q_loss,
            'compute_penalty': compute_penalty,
            'segments_used': num_segments,
            'accuracy': batch_acc
        }
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on a given dataloader."""
        self.model.eval()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        total_segments = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                original_targets = batch['targets'].to(self.device)

                with torch.amp.autocast(device_type='cuda', enabled=self.use_amp):
                    result = self.model(input_ids, max_segments=self.max_segments, training=False)
                    output_logits = result['outputs'][-1]
                    
                    # Correctly shift for causal LM evaluation
                    final_output = output_logits[..., :-1, :].contiguous()
                    targets = original_targets[..., 1:].contiguous()
                    
                    loss = F.cross_entropy(
                        final_output.reshape(-1, final_output.size(-1)),
                        targets.reshape(-1),
                        ignore_index=-100
                    )
                    
                    predictions = final_output.argmax(dim=-1)
                    mask = (targets != -100)
                    correct = (predictions[mask] == targets[mask]).sum().item()
                
                total_loss += loss.item()
                total_correct += correct
                total_samples += mask.sum().item()
                total_segments += result['segments_used']
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': total_correct / total_samples,
            'avg_segments': total_segments / len(dataloader)
        }
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        num_epochs: int,
        save_dir: str = './checkpoints'
    ): # This is a standalone training loop, not used by quick_start_guide.py
        """Main training loop"""
        os.makedirs(save_dir, exist_ok=True)
        best_val_acc = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            self.current_epoch = epoch
            q_weight = self.get_q_loss_weight()
            print(f"Q-loss weight: {q_weight:.3f}")
            epoch_metrics = {
                'task_loss': 0,
                'q_loss': 0,
                'accuracy': 0,
                'segments_used': 0
            }
            
            self.model.train()
            for batch in tqdm(train_dataloader, desc="Training"):
                metrics = self.train_step(batch, num_epochs)
                for key in epoch_metrics:
                    epoch_metrics[key] += metrics[key]
            
            for key in epoch_metrics:
                epoch_metrics[key] /= len(train_dataloader)
            
            val_metrics = self.evaluate(val_dataloader)
            self.scheduler.step(val_metrics['loss'])
            
            self.training_history['losses'].append(epoch_metrics['task_loss'])
            self.training_history['q_losses'].append(epoch_metrics['q_loss'])
            self.training_history['val_accuracies'].append(val_metrics['accuracy'])
            self.training_history['segments_used'].append(val_metrics['avg_segments'])
            
            # Print progress
            print(f"Train Loss: {epoch_metrics['task_loss']:.4f}")
            print(f"Train Acc: {epoch_metrics['accuracy']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"Avg Segments: {val_metrics['avg_segments']:.2f}")
            
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_accuracy': best_val_acc,
                    'training_history': self.training_history
                }, os.path.join(save_dir, 'best_model.pt'))
            
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'training_history': self.training_history
                }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt'))
    
    def plot_training_history(self):
        """Plots the training and validation metrics collected during training."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        axes[0, 0].plot(self.training_history['losses'])
        axes[0, 0].set_title('Task Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        
        axes[0, 1].plot(self.training_history['q_losses'])
        axes[0, 1].set_title('Q-Learning Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        
        axes[1, 0].plot(self.training_history['val_accuracies'])
        axes[1, 0].set_title('Validation Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        
        axes[1, 1].plot(self.training_history['segments_used'])
        axes[1, 1].set_title('Average Segments Used')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Segments')
        
        plt.tight_layout() 
        plt.show()

# Utility functions for model analysis
def analyze_reasoning_steps(model, input_ids, tokenizer, max_segments=8):
    """Analyze the reasoning steps of HRM"""
    model.eval()
    
    with torch.no_grad():
        input_ids = input_ids.to(next(model.parameters()).device)
        result = model(input_ids.unsqueeze(0), max_segments=max_segments, training=False)
    
    print(f"Input: {tokenizer.decode(input_ids)}")
    print(f"Used {result['segments_used']} segments")
    
    for i, (output, q_vals) in enumerate(zip(result['outputs'], result['q_values'])):
        prediction = output.argmax(dim=-1).squeeze()
        decoded = tokenizer.decode(prediction)
        halt_prob = q_vals[0, 0].item()
        
        print(f"\nSegment {i+1}:")
        print(f"  Prediction: {decoded}")
        print(f"  Halt probability: {halt_prob:.3f}")

def compare_with_baseline(hrm_model, baseline_model, test_dataloader, device):
    """Compares HRM accuracy with a baseline Transformer on a test dataloader."""
    hrm_model.eval()
    baseline_model.eval()
    
    hrm_correct = 0
    baseline_correct = 0
    total_samples = 0
    hrm_segments = 0
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Comparing models"):
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)
            
            hrm_result = hrm_model(input_ids, training=False)
            hrm_pred = hrm_result['outputs'][-1].argmax(dim=-1)
            hrm_correct += (hrm_pred == targets).sum().item()
            hrm_segments += hrm_result['segments_used']
            
            baseline_output = baseline_model(input_ids)
            baseline_pred = baseline_output.argmax(dim=-1)
            baseline_correct += (baseline_pred == targets).sum().item()
            
            total_samples += targets.numel()
    
    hrm_acc = hrm_correct / total_samples
    baseline_acc = baseline_correct / total_samples
    avg_segments = hrm_segments / len(test_dataloader)
    
    print(f"HRM Accuracy: {hrm_acc:.4f}")
    print(f"Baseline Accuracy: {baseline_acc:.4f}")
    if baseline_acc > 0:
        print(f"HRM Improvement: {((hrm_acc - baseline_acc) / baseline_acc * 100):+.2f}%")
    else:
        print(f"HRM Improvement: {(hrm_acc - baseline_acc):+.4f} (baseline acc is 0)")
    print(f"Average segments used: {avg_segments:.2f}")
    
    return {
        'hrm_accuracy': hrm_acc,
        'baseline_accuracy': baseline_acc,
        'improvement': hrm_acc - baseline_acc,
        'avg_segments': avg_segments
    }

if __name__ == "__main__":
    print("🧠 HRM Training Module")
    print("Use quick_start_guide.py to run experiments with different configurations.")