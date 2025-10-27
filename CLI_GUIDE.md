# Command-Line Interface Summary

## Quick Reference

### 1. Train Base Model Only

**Purpose**: Train a recommendation model and evaluate its performance without unlearning.

**Command**:
```bash
python train_and_evaluate.py --mode train --dataset movielens-1m --model lightgcn --save_model
```

**What happens**:
- Downloads/loads the dataset
- Creates a base GNN model (LightGCN/SGL/SimGCL)
- Pre-trains the Influence Encoder with simulated unlearning
- Evaluates on test set (Recall@K, NDCG@K)
- Saves checkpoint in `checkpoints/`

**Use cases**:
- Baseline model evaluation
- Compare different models/datasets
- Save model for later unlearning experiments

---

### 2. Train and Unlearn (Both Phases)

**Purpose**: Complete pipeline - train a model, then simulate unlearning and evaluate.

**Command**:
```bash
python train_and_evaluate.py --mode both --dataset movielens-1m --model lightgcn --save_model
```

**What happens**:
1. **Phase 1 - Training**: Same as mode=train
2. **Phase 2 - Unlearning**:
   - Samples edges to unlearn (default 10% of training set)
   - Applies Influence Encoder to remove those edges
   - Optionally fine-tunes
   - Evaluates unlearning effectiveness (score reduction, comparison with negatives)
   - Evaluates model utility (Recall@K, NDCG@K on test set)
   - Saves results in `results/`

**Use cases**:
- Full evaluation of the UnlearnRec framework
- Compare unlearning effectiveness across datasets
- Benchmark different models

---

### 3. Unlearn Only (Load from Checkpoint)

**Purpose**: Perform unlearning on a pre-trained model.

**Command**:
```bash
python train_and_evaluate.py --mode unlearn --load_checkpoint checkpoints/model.pt
```

**What happens**:
- Loads pre-trained model and Influence Encoder
- Performs unlearning (Phase 2 only)
- Evaluates effectiveness and utility

**Use cases**:
- Test different unlearning scenarios on the same base model
- Quick experiments without re-training
- Production deployment (train once, unlearn many times)

---

## Key Parameters

### Dataset Selection
```bash
--dataset movielens-1m    # MovieLens 1M (default)
--dataset gowalla          # Gowalla check-ins
--dataset yelp2018         # Yelp 2018
```

### Model Selection
```bash
--model lightgcn    # LightGCN (default, simple and effective)
--model sgl         # SGL (with self-supervised learning)
--model simgcl      # SimGCL (simplified contrastive learning)
```

### Training Control
```bash
--num_pretrain_epochs 100     # Number of pre-training epochs
--unlearn_ratio 0.05          # Ratio of edges to simulate unlearning during training
--learning_rate 0.001         # Learning rate
```

### Unlearning Control
```bash
--unlearn_test_ratio 0.1      # Ratio of edges to unlearn in test phase
--fine_tune                   # Enable fine-tuning after unlearning
--fine_tune_epochs 5          # Number of fine-tuning epochs
--lambda_u 1.0                # Weight for unlearning loss
--lambda_p 1.0                # Weight for preserving loss
--lambda_c 0.01               # Weight for contrast loss
```

### Evaluation
```bash
--eval_k 20          # K for Recall@K and NDCG@K
--test_ratio 0.2     # Ratio of test set
```

---

## Common Workflows

### Workflow 1: Quick Test
Test the system with minimal compute:

```bash
python train_and_evaluate.py \
    --mode both \
    --dataset movielens-1m \
    --model lightgcn \
    --num_pretrain_epochs 10 \
    --unlearn_test_ratio 0.05 \
    --eval_k 10
```

### Workflow 2: Full Experiment
Run complete training and evaluation:

```bash
python train_and_evaluate.py \
    --mode both \
    --dataset movielens-1m \
    --model lightgcn \
    --num_pretrain_epochs 100 \
    --unlearn_test_ratio 0.1 \
    --fine_tune \
    --fine_tune_epochs 5 \
    --eval_k 20 \
    --save_model
```

### Workflow 3: Compare Models
Benchmark all three models:

```bash
# LightGCN
python train_and_evaluate.py --mode both --dataset movielens-1m --model lightgcn --save_model

# SGL
python train_and_evaluate.py --mode both --dataset movielens-1m --model sgl --save_model

# SimGCL
python train_and_evaluate.py --mode both --dataset movielens-1m --model simgcl --save_model
```

### Workflow 4: Ablation Study on Unlearning Ratios
Test different amounts of unlearning:

```bash
# 5% unlearning
python train_and_evaluate.py --mode both --unlearn_test_ratio 0.05 --save_model

# 10% unlearning
python train_and_evaluate.py --mode both --unlearn_test_ratio 0.10 --save_model

# 20% unlearning
python train_and_evaluate.py --mode both --unlearn_test_ratio 0.20 --save_model
```

### Workflow 5: Use Configuration File
Create reusable configurations:

```bash
# Quick test
python train_and_evaluate.py --config configs/quick_test.yaml --dataset movielens-1m

# Full experiment
python train_and_evaluate.py --config configs/full_experiment.yaml --dataset gowalla
```

---

## Output Interpretation

### Phase 1 Output (Training)
```
Base Model Metrics:
  Recall@20: 0.1234
  NDCG@20: 0.0876
```
- **Recall@20**: 12.34% of relevant items appear in top-20 recommendations
- **NDCG@20**: Ranking quality score (higher is better, max 1.0)

### Phase 2 Output (Unlearning)
```
Unlearning Effectiveness Metrics
Score Reduction: 72.78%
Unlearned/Negative Score Ratio: 1.2503

Model Utility Metrics (on Test Set)
Recall@20: 0.1189
NDCG@20: 0.0842

Embedding Similarity: 0.8934
```

**Unlearning Effectiveness**:
- **Score Reduction**: 72.78% means unlearned edges' scores dropped by ~73% ✓ Good!
- **Unlearned/Negative Ratio**: 1.25 means unlearned edges scored 1.25× higher than random negatives
  - Goal: < 1.0 (unlearned should look like never existed)
  - 1.25 is acceptable, some residual information remains

**Model Utility**:
- **Recall@20**: 0.1189 vs baseline 0.1234 → 3.6% drop (acceptable)
- **NDCG@20**: 0.0842 vs baseline 0.0876 → 3.9% drop (acceptable)
- Trade-off: Effective unlearning with minimal utility loss

**Embedding Similarity**:
- 0.8934 means embeddings are 89% similar to original
- Shows model didn't change drastically (preserving knowledge)

---

## Tips for Best Results

1. **Start with quick tests** (`--num_pretrain_epochs 10`) to verify setup
2. **Use config files** for reproducible experiments
3. **Enable fine-tuning** (`--fine_tune`) for better unlearning
4. **Monitor both metrics**: Unlearning effectiveness AND model utility
5. **Try different models**: LightGCN is fast, SGL/SimGCL may be more robust
6. **Adjust lambda weights**: Increase `--lambda_u` for stronger unlearning, `--lambda_p` to preserve more

---

## Files Generated

### Checkpoints (`checkpoints/`)
```
movielens-1m_lightgcn_base_model_20250428_120000.pt
movielens-1m_lightgcn_unlearned_model_20250428_121500.pt
```
Contains: model weights, Influence Encoder, config, metrics

### Results (`results/`)
```
movielens-1m_lightgcn_results_20250428_121500.json
```
Contains: complete metrics in JSON format

---

## Next Steps

After getting familiar with the CLI:

1. **Tune hyperparameters**: Adjust `lambda_u`, `lambda_p`, `lambda_c` for your use case
2. **Custom datasets**: Add new loaders in `data/preprocessor.py`
3. **Custom models**: Extend `BaseGNN` in `models/`
4. **Advanced evaluation**: Add metrics in `utils/metrics.py`
5. **Production deployment**: Use `--mode unlearn` with saved checkpoints

For more details, see [USAGE.md](USAGE.md).
