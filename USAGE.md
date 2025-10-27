# UnlearnRec Usage Guide

This guide explains how to use the command-line interface for training and evaluating the UnlearnRec recommendation system.

## Quick Start

### 1. Train and Evaluate Base Model Only

Train a LightGCN model on MovieLens-1M and evaluate its performance:

```bash
python train_and_evaluate.py --mode train --dataset movielens-1m --model lightgcn --save_model
```

### 2. Train Base Model and Perform Unlearning

Complete pipeline: train the model, then simulate unlearning:

```bash
python train_and_evaluate.py --mode both --dataset movielens-1m --model lightgcn --save_model
```

### 3. Unlearning Only (Load from Checkpoint)

If you already have a trained model, perform unlearning only:

```bash
python train_and_evaluate.py --mode unlearn --dataset movielens-1m --load_checkpoint checkpoints/movielens-1m_lightgcn_base_model_20250428_120000.pt
```

## Command-Line Arguments

### Mode Selection

- `--mode` : Choose operation mode
  - `train` : Train base model only and evaluate
  - `unlearn` : Perform unlearning only (requires checkpoint)
  - `both` : Train base model then perform unlearning (default)

### Dataset Options

- `--dataset` : Dataset to use (default: `movielens-1m`)
  - `movielens-1m` : MovieLens 1M dataset
  - `gowalla` : Gowalla check-in dataset
  - `yelp2018` : Yelp 2018 dataset
- `--data_dir` : Directory for dataset storage (default: `data`)
- `--test_ratio` : Ratio of test set (default: `0.2`)

### Model Options

- `--model` : Base GNN model (default: `lightgcn`)
  - `lightgcn` : LightGCN model
  - `sgl` : Self-supervised Graph Learning (SGL)
  - `simgcl` : SimGCL model
- `--embedding_dim` : Embedding dimension (default: `64`)
- `--num_layers` : Number of GNN layers (default: `3`)
- `--num_layers_ie` : Number of Influence Encoder layers (default: `3`)
- `--num_layers_mlp` : Number of MLP layers in Influence Encoder (default: `2`)

### Training Options

- `--num_pretrain_epochs` : Number of pre-training epochs (default: `100`)
- `--learning_rate` : Learning rate (default: `0.001`)
- `--batch_size` : Batch size (default: `512`)
- `--unlearn_ratio` : Ratio of edges to unlearn during pre-training (default: `0.05`)

### Unlearning Options

- `--lambda_u` : Weight for unlearning loss (default: `1.0`)
- `--lambda_p` : Weight for preserving loss (default: `1.0`)
- `--lambda_c` : Weight for contrast loss (default: `0.01`)
- `--fine_tune` : Enable fine-tuning after unlearning
- `--fine_tune_epochs` : Number of fine-tuning epochs (default: `3`)
- `--unlearn_test_ratio` : Ratio of edges to unlearn in test phase (default: `0.1`)

### Evaluation Options

- `--eval_k` : K for Recall@K and NDCG@K metrics (default: `20`)

### I/O Options

- `--config` : Path to YAML config file (overrides CLI args)
- `--checkpoint_dir` : Directory to save checkpoints (default: `checkpoints`)
- `--results_dir` : Directory to save results (default: `results`)
- `--save_model` : Save trained models
- `--load_checkpoint` : Path to checkpoint to load

### Other Options

- `--seed` : Random seed (default: `42`)
- `--device` : Device to use: `auto`, `cuda`, or `cpu` (default: `auto`)

## Example Workflows

### Example 1: Quick Test on MovieLens

Train LightGCN with minimal epochs for quick testing:

```bash
python train_and_evaluate.py \
    --mode both \
    --dataset movielens-1m \
    --model lightgcn \
    --num_pretrain_epochs 10 \
    --eval_k 10 \
    --save_model
```

### Example 2: Full Training on Gowalla with SGL

Train SGL model on Gowalla dataset with full settings:

```bash
python train_and_evaluate.py \
    --mode both \
    --dataset gowalla \
    --model sgl \
    --embedding_dim 128 \
    --num_layers 3 \
    --num_pretrain_epochs 100 \
    --learning_rate 0.001 \
    --unlearn_ratio 0.05 \
    --fine_tune \
    --fine_tune_epochs 5 \
    --eval_k 20 \
    --save_model
```

### Example 3: Using Configuration File

Create a config file `my_config.yaml`:

```yaml
# Model parameters
embedding_dim: 128
num_layers: 4
num_layers_ie: 3
num_layers_mlp: 2

# Loss weights
lambda_u: 1.5
lambda_p: 1.0
lambda_c: 0.01

# Training parameters
num_pretrain_epochs: 200
learning_rate: 0.0005
unlearn_ratio: 0.05
fine_tune: true
fine_tune_epochs: 5

# Evaluation
eval_k: 20
test_ratio: 0.2
```

Then run:

```bash
python train_and_evaluate.py \
    --mode both \
    --dataset yelp2018 \
    --model simgcl \
    --config my_config.yaml \
    --save_model
```

### Example 4: Train Base Model, Then Unlearn Later

Step 1: Train and save base model

```bash
python train_and_evaluate.py \
    --mode train \
    --dataset movielens-1m \
    --model lightgcn \
    --num_pretrain_epochs 100 \
    --save_model
```

Step 2: Perform unlearning from saved checkpoint

```bash
python train_and_evaluate.py \
    --mode unlearn \
    --dataset movielens-1m \
    --model lightgcn \
    --load_checkpoint checkpoints/movielens-1m_lightgcn_base_model_TIMESTAMP.pt \
    --unlearn_test_ratio 0.1 \
    --fine_tune \
    --save_model
```

### Example 5: Compare Different Unlearning Ratios

Test different unlearning ratios to see effectiveness:

```bash
# 5% unlearning
python train_and_evaluate.py --mode both --dataset movielens-1m --unlearn_test_ratio 0.05 --save_model

# 10% unlearning
python train_and_evaluate.py --mode both --dataset movielens-1m --unlearn_test_ratio 0.10 --save_model

# 20% unlearning
python train_and_evaluate.py --mode both --dataset movielens-1m --unlearn_test_ratio 0.20 --save_model
```

## Understanding the Output

### Phase 1: Training Base Model

```
PHASE 1: Training Base Recommendation Model
================================================================================
Model: lightgcn
Users: 6040, Items: 3706
Total interactions: 1000209
Train edges: 800167, Test edges: 200042

Starting pre-training...
Pre-training: 100%|██████████| 100/100 [01:23<00:00, 1.20it/s]
Epoch 0: Total Loss: 2.4567
Epoch 10: Total Loss: 1.8234
...

--------------------------------------------------------------------------------
Evaluating Base Model Performance
--------------------------------------------------------------------------------
Base Model Metrics:
  Recall@20: 0.1234
  NDCG@20: 0.0876
```

### Phase 2: Unlearning and Evaluation

```
PHASE 2: Unlearning and Evaluation
================================================================================
Unlearning 8000 edges (10.0% of training set)
Remaining edges: 792167

Average original score for unlearn edges: 0.4532

Processing unlearning request...
Average unlearned score for unlearn edges: 0.1234
Average score for negative edges: 0.0987

--------------------------------------------------------------------------------
Unlearning Effectiveness Metrics
--------------------------------------------------------------------------------
Score Reduction: 72.78%
Unlearned/Negative Score Ratio: 1.2503
  (Lower is better, <1.0 means unlearned edges scored below random negatives)

--------------------------------------------------------------------------------
Model Utility Metrics (on Test Set)
--------------------------------------------------------------------------------
Recall@20: 0.1189
NDCG@20: 0.0842

Embedding Similarity (Original vs Unlearned): 0.8934
```

### Key Metrics Explained

**Unlearning Effectiveness:**
- **Score Reduction**: Percentage decrease in prediction scores for unlearned edges (higher is better)
- **Unlearned/Negative Ratio**: Ratio of unlearned scores to random negative scores (lower is better, <1.0 is ideal)

**Model Utility:**
- **Recall@K**: Fraction of relevant items in top-K recommendations (higher is better)
- **NDCG@K**: Normalized Discounted Cumulative Gain (higher is better)

**Embedding Similarity**: Cosine similarity between original and unlearned embeddings (measures how much the model changed)

## Output Files

### Checkpoints

Saved in `checkpoints/` directory:
- Format: `{dataset}_{model}_{suffix}_{timestamp}.pt`
- Contains: model weights, influence encoder weights, configuration, metrics

### Results

Saved in `results/` directory:
- Format: `{dataset}_{model}_results_{timestamp}.json`
- Contains: complete configuration, all metrics in JSON format

## Tips

1. **Start Small**: Use `--num_pretrain_epochs 10` for quick tests before full training
2. **GPU Usage**: The script auto-detects GPU; force CPU with `--device cpu` if needed
3. **Memory Issues**: Reduce `--batch_size` or `--embedding_dim` if you run out of memory
4. **Reproducibility**: Use `--seed` to ensure reproducible results
5. **Hyperparameter Tuning**: Use config files to organize different experimental setups

## Troubleshooting

### Dataset Download Issues

If automatic download fails:

```bash
# Create data directory
mkdir data

# For MovieLens-1M:
# Download ml-1m.zip from https://grouplens.org/datasets/movielens/1m/
# Place in data/ml-1m.zip

# For Gowalla:
# Download from https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz
# Place in data/gowalla_checkins.txt.gz

# For Yelp2018:
# Download yelp2018.csv and place in data/yelp2018.csv
```

### CUDA Out of Memory

Reduce memory usage:

```bash
python train_and_evaluate.py \
    --embedding_dim 32 \
    --batch_size 256 \
    --num_layers 2
```

### Slow Training

Use smaller epochs for testing:

```bash
python train_and_evaluate.py \
    --num_pretrain_epochs 10 \
    --eval_k 10
```
