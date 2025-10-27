# UnlearnRec Quick Start Cheat Sheet

## Installation
```bash
pip install -r requirements.txt
```

## Three Ways to Run

### 1️⃣ Interactive Menu (Easiest)
```bash
# Windows
run.bat

# Linux/Mac
chmod +x run.sh && ./run.sh
```

### 2️⃣ Command Line (Most Flexible)
```bash
# Train base model only
python train_and_evaluate.py --mode train --dataset movielens-1m --save_model

# Train and unlearn
python train_and_evaluate.py --mode both --dataset movielens-1m --save_model

# Unlearn from checkpoint
python train_and_evaluate.py --mode unlearn --load_checkpoint checkpoints/model.pt
```

### 3️⃣ Config File (Most Reproducible)
```bash
python train_and_evaluate.py --config configs/quick_test.yaml --dataset movielens-1m
```

---

## Most Common Commands

### Quick Test (2-3 min)
```bash
python train_and_evaluate.py --mode both --num_pretrain_epochs 10
```

### Full Training (30-60 min)
```bash
python train_and_evaluate.py --mode both --num_pretrain_epochs 100 --fine_tune --save_model
```

### Different Datasets
```bash
# MovieLens-1M (default)
python train_and_evaluate.py --mode both --dataset movielens-1m

# Gowalla
python train_and_evaluate.py --mode both --dataset gowalla

# Yelp2018
python train_and_evaluate.py --mode both --dataset yelp2018
```

### Different Models
```bash
# LightGCN (default, fastest)
python train_and_evaluate.py --mode both --model lightgcn

# SGL (self-supervised)
python train_and_evaluate.py --mode both --model sgl

# SimGCL (contrastive)
python train_and_evaluate.py --mode both --model simgcl
```

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mode` | `both` | `train`, `unlearn`, or `both` |
| `--dataset` | `movielens-1m` | Dataset to use |
| `--model` | `lightgcn` | GNN model type |
| `--num_pretrain_epochs` | `100` | Training epochs |
| `--unlearn_test_ratio` | `0.1` | % of edges to unlearn |
| `--eval_k` | `20` | K for Recall@K, NDCG@K |
| `--save_model` | `False` | Save checkpoints |
| `--fine_tune` | `False` | Fine-tune after unlearning |

---

## Output Locations

| What | Where | Example |
|------|-------|---------|
| Checkpoints | `checkpoints/` | `movielens-1m_lightgcn_base_model_20250428.pt` |
| Results | `results/` | `movielens-1m_lightgcn_results_20250428.json` |
| Data | `data/` | `movielens_1m_interactions.csv` |

---

## Understanding Output

### Good Unlearning ✓
```
Score Reduction: 70%+           ← High is good
Unlearned/Negative Ratio: <1.5  ← Low is good
Recall@20: >0.10                ← Depends on dataset
```

### What Each Metric Means
- **Score Reduction**: % drop in scores for unlearned edges (higher = better unlearning)
- **Unlearned/Negative Ratio**: How unlearned edges compare to random negatives (lower = better)
- **Recall@K**: % of relevant items in top-K recommendations (higher = better utility)
- **NDCG@K**: Ranking quality (higher = better utility)

---

## Common Issues

### Out of Memory
```bash
python train_and_evaluate.py --embedding_dim 32 --batch_size 256
```

### Too Slow
```bash
python train_and_evaluate.py --num_pretrain_epochs 10 --num_layers 2
```

### Dataset Not Downloading
```bash
# Download manually and place in data/ folder
# See USAGE.md for URLs
```

---

## Example Experiments

### Baseline
```bash
python train_and_evaluate.py --mode both --save_model
```

### Tune Unlearning Strength
```bash
python train_and_evaluate.py --mode both --lambda_u 2.0 --save_model
```

### More Aggressive Unlearning
```bash
python train_and_evaluate.py --mode both --unlearn_test_ratio 0.2 --fine_tune --save_model
```

### Compare Models
```bash
python train_and_evaluate.py --mode both --model lightgcn --save_model
python train_and_evaluate.py --mode both --model sgl --save_model
python train_and_evaluate.py --mode both --model simgcl --save_model
```

---

## Full Documentation

- **README.md**: Project overview
- **USAGE.md**: Complete reference with examples
- **CLI_GUIDE.md**: Quick reference for common tasks
- **IMPLEMENTATION_SUMMARY.md**: What was implemented and why

---

## One-Line Summary

**Train a model**: `python train_and_evaluate.py --mode train --save_model`

**Train and unlearn**: `python train_and_evaluate.py --mode both --save_model`

That's it! 🚀
