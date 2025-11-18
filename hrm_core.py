import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True) * (x.size(-1) ** -0.5)
        return self.weight * x / (norm + self.eps)

class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x, seq_len: int):
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_emb = emb.cos()
        sin_emb = emb.sin()
        return cos_emb, sin_emb

def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding to query and key tensors"""
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class MultiHeadAttention(nn.Module):
    """Multi-head attention with Rotary Position Embedding (RoPE)."""
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        self.rope = RotaryPositionalEmbedding(self.head_dim)
        
    def forward(self, x):
        B, T, C = x.shape
        
        # Get Q, K, V
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2), qkv)
        
        # Apply RoPE
        cos, sin = self.rope(x, T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.proj(out)

class SwiGLU(nn.Module):
    """SwiGLU activation function"""
    def __init__(self, dim: int, expansion: float = 2.0):
        super().__init__()
        hidden_dim = int(dim * expansion)
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        
    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class TransformerBlock(nn.Module):
    """A standard Transformer block using a pre-normalization architecture."""
    def __init__(self, dim: int, num_heads: int, expansion: float = 2.0, dropout: float = 0.1, residual_scale: float = 1.0):
        super().__init__() # Removed gradient_checkpointing from init, will pass as forward arg
        self.attention = MultiHeadAttention(dim, num_heads, dropout)
        self.feed_forward = SwiGLU(dim, expansion)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        
        # Scales the output of residual branches. Helps stabilize deep models.
        self.residual_scale = residual_scale
        
    def forward(self, x, gradient_checkpointing: bool = False):
        if gradient_checkpointing and self.training: # Only checkpoint during training
            return torch.utils.checkpoint.checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)

    def _forward_impl(self, x):
        # Pre-norm architecture: x = x + F(norm(x))
        x = x + self.residual_scale * self.attention(self.norm1(x))
        x = x + self.residual_scale * self.feed_forward(self.norm2(x))
        return x

@dataclass
class HRMCarry:
    """State carrier for HRM"""
    z_H: torch.Tensor
    steps: torch.Tensor
    halted: torch.Tensor

class ReasoningModule(nn.Module):
    """A stack of Transformer blocks, representing a single reasoning module (H or L)."""
    def __init__(self, vocab_size, dim=256, num_heads=8, num_layers=6, 
                 residual_scale=1.0, expansion=2.0, dropout=0.1, gradient_checkpointing: bool = False):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, expansion, dropout, residual_scale=residual_scale) 
            for _ in range(num_layers) # Pass gradient_checkpointing to TransformerBlock
        ])
        
    def forward(self, hidden_states, input_injection=None):
        """
        Forward pass through the reasoning module.
        Args:
            hidden_states: Current hidden state
            input_injection: Optional state to add to the input (e.g., from another module).
        """
        if input_injection is not None:
            hidden_states = hidden_states + input_injection
            
        for layer in self.layers:
            hidden_states = layer(hidden_states, gradient_checkpointing=self.gradient_checkpointing)
            
        return hidden_states

class HierarchicalReasoningModel(nn.Module):
    """
    The Hierarchical Reasoning Model (HRM).
    """
    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        expansion: float = 2.0,
        max_seq_len: int = 1024, # Max sequence length for positional embeddings
        H_cycles: int = 2,      # "Planner" cycles (N in the paper)
        L_cycles: int = 2,      # "Executor" cycles per H-cycle (T in the paper)
        dropout: float = 0.1,
        halt_max_steps: int = 4,
        residual_scale: float = 0.8,
        adaptive_cycles: bool = True,
        gradient_checkpointing: bool = False # Added for gradient checkpointing
    ):
        super().__init__()
        self.dim = dim
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.halt_max_steps = halt_max_steps
        self.adaptive_cycles = adaptive_cycles
        
        self.gradient_checkpointing = gradient_checkpointing
        # Adaptive cycle strategy for efficiency
        # For tiny tasks: H=1, L=1; For complex tasks: H=2-3, L=2-3
        self.min_H_cycles = 1 # Minimum cycles to run, even if adaptive logic suggests fewer.
        self.min_L_cycles = 1
        
        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        
        # The H-module (planner) operates in a higher-dimensional space for more abstract reasoning.
        h_dim = int(dim * 3)
        h_dim = (h_dim // num_heads) * num_heads
        l_dim = dim
        
        self.H_module = ReasoningModule(vocab_size, h_dim, num_heads, num_layers, residual_scale=residual_scale, gradient_checkpointing=gradient_checkpointing)
        self.L_module = ReasoningModule(vocab_size, l_dim, num_heads, num_layers, residual_scale=residual_scale, gradient_checkpointing=gradient_checkpointing)
        
        # Projection layers to handle dimensionality differences
        self.h_to_l_proj = nn.Linear(h_dim, l_dim)
        self.l_to_h_proj = nn.Linear(l_dim, h_dim)
        self.h_to_output_proj = nn.Linear(h_dim, dim)
        
        # Auxiliary heads for deep supervision at each reasoning segment.
        self.aux_lm_heads = nn.ModuleList([
            nn.Linear(dim, vocab_size, bias=False) for _ in range(max(halt_max_steps - 1, 0))
        ])
        
        # Head to predict halting vs. continuing.
        self.q_head = nn.Linear(dim, 2)  # [halt_probability, continue_probability]
        
        # Gating layers for stable state updates
        self.h_gate = nn.Linear(h_dim, h_dim)
        self.l_gate = nn.Linear(l_dim, l_dim)
        self.h_feedback_gate = nn.Linear(h_dim, h_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Store tokenizer reference for generation
        self.tokenizer = None
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, input_ids, max_segments=None, training=True):
        """
        Forward pass for the HRM model.
        Args:
            input_ids: Input token IDs [batch_size, seq_len].
            max_segments: Maximum number of reasoning segments to run (for ACT).
            training: Boolean, true if in training mode.
            
        Returns:
            Dictionary with outputs and metadata
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        x_embed = self.embed_tokens(input_ids) * math.sqrt(self.dim)
        
        # Initialize states for H and L modules.
        z_L = x_embed.clone()
        z_H = self.l_to_h_proj(x_embed)
        
        steps = torch.zeros(batch_size, dtype=torch.long, device=device)
        halted = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        outputs = []
        q_values = []
        
        max_segments = max_segments or self.halt_max_steps
        
        for segment in range(max_segments):
            # During inference, optionally adapt the number of cycles based on confidence.
            if self.adaptive_cycles and not training:
                # During inference, use adaptive cycles based on confidence
                current_H_cycles = self._get_adaptive_H_cycles(z_H, segment)
                current_L_cycles = self._get_adaptive_L_cycles(z_L, segment)
            else:
                current_H_cycles = self.H_cycles
                current_L_cycles = self.L_cycles

            # Initial injection of the input embedding
            z_L = z_L + x_embed
            z_H = z_H + self.l_to_h_proj(x_embed)

            for h_cycle in range(current_H_cycles):
                # H-module provides guidance to L-module
                h_guidance = self.h_to_l_proj(z_H)
                l_gate_values = torch.sigmoid(self.l_gate(h_guidance))
                z_L = l_gate_values * z_L + (1 - l_gate_values) * h_guidance
                
                for l_cycle in range(current_L_cycles):
                    z_L = self.L_module(z_L) # No more injection inside the loop
                
                # The H-module (planner) receives feedback from the L-module (executor).
                l_feedback = self.l_to_h_proj(z_L)
                z_H = self.H_module(z_H, input_injection=l_feedback)
            
            # Project H-state to output dimension and generate logits.
            h_output = self.h_to_output_proj(z_H)
            if segment < len(self.aux_lm_heads):
                logits = self.aux_lm_heads[segment](h_output)
            else:
                logits = self.lm_head(h_output)
            outputs.append(logits)

            # Predict halt/continue probabilities from the pooled H-state.
            pooled_h_state = self.h_to_output_proj(z_H.mean(dim=1))
            q_vals = torch.sigmoid(self.q_head(pooled_h_state))
            q_values.append(q_vals)
            
            steps = steps + (~halted).long()
            
            # During inference, use the Q-values to decide whether to halt.
            if not training and segment >= 1:
                halt_prob = q_vals[:, 0]
                continue_prob = q_vals[:, 1]
                
                # Halt if the predicted halt probability is higher than continue probability.
                should_halt = halt_prob > continue_prob
                
                # Add a confidence threshold for stability.
                confidence_threshold = 0.8
                high_confidence = torch.max(halt_prob, continue_prob) > confidence_threshold
                should_halt = should_halt & high_confidence
                halted = halted | should_halt
                
                # If all samples have halted, break
                if halted.all():
                    break
        
        return {
            'outputs': outputs,
            'q_values': q_values,
            'segments_used': len(outputs),
            'final_states': {'z_H': z_H, 'z_L': z_L}
        }
    
    def _get_adaptive_H_cycles(self, z_H: torch.Tensor, segment: int) -> int:
        """Dynamically determine H-cycles based on model confidence."""
        if segment == 0:
            return max(self.min_H_cycles, self.H_cycles)
        
        # Use halt probability as a proxy for confidence.
        pooled_h_state = self.h_to_output_proj(z_H.mean(dim=1))
        confidence = torch.sigmoid(self.q_head(pooled_h_state)[:, 0]).mean().item()
        
        if confidence > 0.8:  # High confidence - less planning needed
            return self.min_H_cycles
        elif confidence > 0.6:  # Medium confidence
            return max(self.min_H_cycles, self.H_cycles - 1)
        else:  # Low confidence - more planning needed
            return self.H_cycles
    
    def _get_adaptive_L_cycles(self, z_L: torch.Tensor, segment: int) -> int:
        """Dynamically determine L-cycles."""
        if segment == 0:
            return max(self.min_L_cycles, self.L_cycles)
        
        return max(self.min_L_cycles, self.L_cycles)
    
    def generate(self, input_ids, max_length=256, temperature=0.8):
        """Generate a sequence of tokens autoregressively."""
        self.eval()
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        generated = input_ids.clone()
        
        for _ in range(max_length - seq_len):
            with torch.no_grad():
                result = self.forward(generated, max_segments=4, training=False)
                logits = result['outputs'][-1][:, -1, :] # Logits for the last token
                
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                generated = torch.cat([generated, next_token], dim=-1)
                
                if self.tokenizer and next_token.item() in [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]:
                    break
        
        return generated

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Example usage and test for the HRM model.
    model = HierarchicalReasoningModel(
        vocab_size=50257,
        dim=128,  # Smaller for testing
        num_heads=4,
        num_layers=1,
        H_cycles=2,  # Planner cycles
        L_cycles=2,  # Executor cycles
        halt_max_steps=3,
        adaptive_cycles=True
    )
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, 50257, (batch_size, seq_len))
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    result = model(input_ids, max_segments=2, training=True)
    loss = F.cross_entropy(
        result['outputs'][-1].view(-1, result['outputs'][-1].size(-1)),
        input_ids.view(-1)
    )
    
    optimizer.zero_grad()
    loss.backward()
    
    total_norm = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    print(f"Gradient norm: {total_norm:.6f}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Segments used: {result['segments_used']}")
    
    if total_norm > 0:
        print("✅ Gradient flow is working!")
    else:
        print("❌ Gradient flow blocked")