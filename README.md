# Hierarchical Reasoning Model (HRM) vs Baseline Transformer Model

A novel transformer architecture that outperforms standard transformers on complex reasoning tasks through explicit hierarchical planning and execution.

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

The **Hierarchical Reasoning Model (HRM)** is a groundbreaking transformer architecture that introduces explicit hierarchical reasoning through planner-executor modules and iterative reasoning cycles. This project demonstrates how HRM significantly outperforms parameter-matched baseline transformers on complex code generation and reasoning tasks.


## Features

### Architectural Innovations
- **Planner-Executor Architecture**: H-module (planner) for high-level strategy, L-module (executor) for detailed implementation
- **Iterative Reasoning Cycles**: Multi-step internal reasoning before final prediction
- **Adaptive Computation Time**: Dynamic reasoning depth based on task complexity
- **Q-Learning Halting**: Reinforcement learning for optimal stopping decisions
- **Memory Optimization**: Gradient checkpointing and mixed precision training

### Technical Capabilities
- **Parameter-matched comparisons** with baseline transformers
- **Comprehensive evaluation** on HumanEval and synthetic coding tasks
- **Progressive training** with curriculum learning and deep supervision
- **Production-ready** with FastAPI integration potential

## Architecture

### HRM Core Components
Input → H-Module (Planner) → L-Module (Executor) → Output
↑ ↓
Feedback Loops Multiple Cycles


**Key Differences from Standard Transformers:**
| Aspect | HRM | Baseline Transformer |
|--------|-----|---------------------|
| **Reasoning** | Multi-step hierarchical | Single-pass flat |
| **Planning** | Explicit H-cycles | Implicit via attention |
| **Execution** | Multiple L-cycles | Single forward pass |
| **Halting** | Learned via Q-learning | Fixed depth |
| **Efficiency** | Adaptive computation | Static compute |

## Performance Results

| Experiment | HRM Accuracy | Baseline Accuracy | Improvement |
|------------|--------------|-------------------|-------------|
| Tiny (16M params) | 44.58% | 43.72% | +0.86% |
| Small (36M params) | 85.34% | 67.26% | +18.08%|
| Medium (64M params) | 94.17% | 70.37% | +23.80% |

## How to use

```bash
# Clone repository
git clone https://github.com/Prishatank0607/Hierarchical-Reasoning-Model.git
cd Hierarchical-Reasoning-Model

# Install dependencies
pip install -r requirements.txt

# Verify installation
python quick_start_guide.py
Quick Start

1. Run Quick Test

bash
python quick_start_guide.py

2. Run Experiments

python
# Choose from TINY, SMALL, or MEDIUM experiments
python quick_start_guide.py
# Follow the interactive prompts to select experiment size
```

## Project Structure

Hierarchical-Reasoning-Model/
├── hrm_core.py              # Core HRM architecture
├── hrm_training.py          # Advanced training pipeline
├── coding_dataset.py        # Data loading & preprocessing
├── evaluate_coding.py       # Comprehensive evaluation
├── quick_start_guide.py     # Main experiment runner
├── updated_training_loop.py # Alternative training script
└── requirements.txt         # Dependencies

## Key Modules

### Core Architecture (hrm_core.py)
HierarchicalReasoningModel: Main HRM class
ReasoningModule: Planner and executor modules
Advanced components: Rotary embeddings, SwiGLU, RMSNorm

### Training Pipeline (hrm_training.py)
HRMTrainer: Advanced training with curriculum learning
Q-learning integration for adaptive halting
Deep supervision and gradient optimization

### Evaluation Suite (evaluate_coding.py)
HumanEval benchmark integration
Code execution and testing
Comprehensive comparison metrics

## Experiment Configurations

The project includes three pre-configured experiment sizes:

TINY (16M params): Quick validation
SMALL (36M params): Balanced performance
MEDIUM (64M params): Maximum accuracy

## Research Insights

### Proven Advantages
Architectural Efficiency: HRM achieves better performance with similar parameters
Scalable Reasoning: Performance gap widens with task complexity
Interpretable Reasoning: Transparent planning and execution cycles

### Technical Innovations
Hierarchical Convergence: Prevents reasoning stalls
Approximate Gradient Method: Enables deep recursion
Adaptive Computational Time: Dynamic resource allocation

