# Hierarchical Reasoning Model (HRM) vs Baseline Transformer Model

A novel transformer architecture that outperforms standard transformers on complex reasoning tasks through explicit hierarchical planning and execution.

## Overview

The Hierarchical Reasoning Model (HRM) introduces a new approach to transformer-based reasoning by incorporating explicit hierarchical planning, multi-step reasoning cycles, and adaptive halting. Unlike standard transformers that make predictions in a single pass, HRM uses planner–executor modules to break down complex tasks and achieve significantly improved accuracy on reasoning and code-generation benchmarks.

## Features

### Architectural Innovations
- Planner–Executor Architecture  
  - H-module (Planner) for high-level strategy  
  - L-module (Executor) for detailed execution  
- Multi-cycle iterative reasoning  
- Adaptive Computation Time (ACT)  
- Q-learning-based halting mechanism  
- Memory-efficient training with gradient checkpointing and mixed precision

### Technical Capabilities
- Parameter-matched comparison with baseline transformers  
- Evaluation on HumanEval and synthetic coding tasks  
- Curriculum learning and deep supervision  
- Compatible with FastAPI integration for production deployment

## HRM vs Baseline Transformer

| Aspect | HRM | Baseline Transformer |
|--------|-----|----------------------|
| Reasoning | Multi-step hierarchical | Single-pass |
| Planning | Explicit planning cycles | Implicit attention |
| Execution | Recurrent L-cycles | One forward pass |
| Halting | Q-learning guided | Fixed depth |
| Efficiency | Adaptive compute | Static compute |

## Performance Results

| Experiment | HRM Accuracy | Baseline Accuracy | Improvement |
|------------|--------------|-------------------|-------------|
| Tiny (16M params) | 44.58% | 43.72% | +0.86% |
| Small (36M params) | 85.34% | 67.26% | +18.08% |
| Medium (64M params) | 94.17% | 70.37% | +23.80% |

## How to Use

### Clone Repository
```bash
git clone https://github.com/Prishatank0607/Hierarchical-Reasoning-Model.git
cd Hierarchical-Reasoning-Model
```
### Install Dependencies
```
pip install -r requirements.txt
```
### Run Quick Test
```
python quick_start_guide.py
```
### Run Experiments (Tiny, Small, Medium)
```
python quick_start_guide.py
# Select experiment size in interactive mode
```


## Project Structure
Hierarchical-Reasoning-Model/
├── hrm_core.py              # Core HRM architecture
├── hrm_training.py          # Training pipeline with curriculum learning
├── coding_dataset.py        # Dataset loading and preprocessing
├── evaluate_coding.py       # Evaluation suite (HumanEval and synthetic tasks)
├── quick_start_guide.py     # Experiment runner
├── updated_training_loop.py # Alternative recursive training script
└── requirements.txt         # Dependencies

## Key Modules
### Core Architecture (hrm_core.py)
HierarchicalReasoningModel
Planner and executor modules
Rotary embeddings, SwiGLU activation, RMSNorm normalization

### Training Pipeline (hrm_training.py)
HRMTrainer
Q-learning halting mechanism
Curriculum learning and deep supervision

### Evaluation Suite (evaluate_coding.py)
HumanEval integration
Execution-based correctness evaluation
Metric comparison with baseline transformers

## Experiment Configurations
The repository provides three predefined experiment settings:
Tiny (16M parameters): Fast validation
Small (36M parameters): Balanced accuracy and efficiency
Medium (64M parameters): Highest accuracy

## Research Insights

### Proven Advantages
Better performance with similar parameter count
Performance improvements increase with task complexity
Transparent hierarchical reasoning cycles

### Technical Innovations
Hierarchical convergence for stable deep reasoning
Approximate gradient propagation enabling recursion
Adaptive computation time for optimal resource usage
