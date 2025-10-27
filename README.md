# UnlearnRec: Pre-training for Recommendation Unlearning

Implementation of "Pre-training for Recommendation Unlearning" (SIGIR'25).

This repository provides a complete framework for training GNN-based recommender systems with built-in unlearning capabilities using an Influence Encoder.

## Features

- **Multiple GNN Models**: LightGCN, SGL, SimGCL
- **Multiple Datasets**: MovieLens-1M, Gowalla, Yelp2018 (with automatic download)
- **Influence Encoder**: Pre-trained component for efficient unlearning
- **Command-Line Interface**: Easy training and evaluation
- **Comprehensive Metrics**: Unlearning effectiveness and model utility metrics

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd UnlearnRec

# Install dependencies
pip install torch numpy pandas pyyaml scikit-learn tqdm
```

### Basic Usage

#### Option 1: Using the Quick Start Script (Windows)

```bash
run.bat
```

Then select an option from the menu.

#### Option 2: Using the Quick Start Script (Unix/Linux/Mac)

```bash
chmod +x run.sh
./run.sh
```

#### Option 3: Command Line

**Train a base model and evaluate:**

```bash
python train_and_evaluate.py --mode train --dataset movielens-1m --model lightgcn --save_model
```

**Train and perform unlearning:**

```bash
python train_and_evaluate.py --mode both --dataset movielens-1m --model lightgcn --save_model
```

**Unlearn from existing checkpoint:**

```bash
python train_and_evaluate.py --mode unlearn --load_checkpoint checkpoints/model.pt
```

## Detailed Usage

See [USAGE.md](USAGE.md) for comprehensive documentation including:
- All command-line arguments
- Example workflows
- Configuration file usage
- Troubleshooting guide

## Project Structure

```
UnlearnRec/
├── train_and_evaluate.py    # Main CLI tool
├── main.py                   # Simple example script
├── run.bat                   # Quick start script (Windows)
├── run.sh                    # Quick start script (Unix/Linux/Mac)
├── USAGE.md                  # Detailed usage guide
├── configs/
│   ├── base.yaml            # Default configuration
│   └── model_config.yaml    # Model-specific configs
├── core/
│   ├── influence_encoder.py # Influence Encoder implementation
│   ├── loss_function.py     # Multi-task loss functions
│   └── unlearning_manager.py # Unlearning pipeline
├── data/
│   ├── dataset.py           # Dataset classes
│   └── preprocessor.py      # Data loading and preprocessing
├── models/
│   ├── base_gnn.py          # Base GNN class
│   ├── lightgcn.py          # LightGCN implementation
│   ├── sgl.py               # SGL implementation
│   └── simgcl.py            # SimGCL implementation
├── training/
│   ├── pretrainer.py        # Pre-training logic
│   └── fine_tuner.py        # Fine-tuning logic
└── utils/
    ├── config.py            # Configuration utilities
    ├── metrics.py           # Evaluation metrics
    └── visualization.py     # Visualization tools
```

## Supported Datasets

- **MovieLens-1M**: 1 million ratings from 6,040 users on 3,706 movies
- **Gowalla**: Location-based social network check-ins
- **Yelp2018**: User-business interactions from Yelp

Datasets are automatically downloaded on first use.

## Supported Models

- **LightGCN**: Simple and effective GNN for recommendations
- **SGL**: Self-supervised Graph Learning with contrastive learning
- **SimGCL**: Simplified contrastive learning for graphs

## Key Components

### Influence Encoder

Pre-trained component that learns to adjust embeddings for unlearning:
- Trainable parameters: H₀ (initial influence), W_η (transformation)
- Multi-layer architecture with MLP for embedding adjustments
- Frozen after pre-training for efficient inference

### Loss Functions

Multi-task loss combining:
- **L_M**: Model-specific loss (BPR + SSL)
- **L_U**: Unlearning loss (reduce scores for forgotten edges)
- **L_P**: Preserving loss (maintain distribution for remaining edges)
- **L_C**: Contrastive loss (generalization across influence patterns)

### Unlearning Pipeline

1. **Pre-training**: Train Influence Encoder with simulated unlearning requests
2. **Inference**: Apply pre-trained encoder to real unlearning requests
3. **Optional Fine-tuning**: Further optimize for specific requests

## Evaluation Metrics

### Unlearning Effectiveness
- **Score Reduction**: % decrease in prediction scores for unlearned edges
- **Unlearned/Negative Ratio**: Comparison with random negative samples

### Model Utility
- **Recall@K**: Fraction of relevant items in top-K recommendations
- **NDCG@K**: Normalized Discounted Cumulative Gain

## Example Commands

### Quick Test (10 epochs)

```bash
python train_and_evaluate.py \
    --mode both \
    --dataset movielens-1m \
    --model lightgcn \
    --num_pretrain_epochs 10 \
    --save_model
```

### Full Training (100 epochs)

```bash
python train_and_evaluate.py \
    --mode both \
    --dataset movielens-1m \
    --model lightgcn \
    --num_pretrain_epochs 100 \
    --fine_tune \
    --fine_tune_epochs 5 \
    --save_model
```

### Using a Configuration File

```bash
python train_and_evaluate.py \
    --mode both \
    --dataset gowalla \
    --model sgl \
    --config configs/base.yaml \
    --save_model
```

### Compare Different Models

```bash
# LightGCN
python train_and_evaluate.py --mode both --dataset movielens-1m --model lightgcn --save_model

# SGL
python train_and_evaluate.py --mode both --dataset movielens-1m --model sgl --save_model

# SimGCL
python train_and_evaluate.py --mode both --dataset movielens-1m --model simgcl --save_model
```

## Output Files

### Checkpoints
Saved in `checkpoints/`:
- Model weights
- Influence Encoder weights
- Configuration
- Training history

### Results
Saved in `results/`:
- Complete metrics in JSON format
- Timestamp for tracking experiments

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{unlearnrec2025,
  title={Pre-training for Recommendation Unlearning},
  booktitle={Proceedings of the 48th International ACM SIGIR Conference},
  year={2025}
}
```
