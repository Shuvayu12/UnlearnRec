# UnlearnRec CLI - Implementation Summary

## What Was Created

I've implemented a comprehensive command-line interface for training and evaluating the UnlearnRec recommendation system. This allows you to:

1. **Train base recommendation models** with different GNN architectures
2. **Perform unlearning** and evaluate effectiveness
3. **Compare datasets and models** systematically
4. **Save and load checkpoints** for reproducibility

---

## New Files Created

### 1. `train_and_evaluate.py` (Main CLI Tool)
**Purpose**: Complete command-line interface for training and unlearning

**Features**:
- Argument parsing for all hyperparameters
- Three modes: `train`, `unlearn`, `both`
- Support for multiple datasets (MovieLens-1M, Gowalla, Yelp2018)
- Support for multiple models (LightGCN, SGL, SimGCL)
- Automatic checkpoint saving and loading
- Comprehensive metrics evaluation
- Results export to JSON

### 2. `USAGE.md` (Detailed Documentation)
**Purpose**: Complete user guide with examples

**Contents**:
- Quick start guide
- All command-line arguments explained
- Example workflows for different scenarios
- Output interpretation
- Troubleshooting guide

### 3. `CLI_GUIDE.md` (Quick Reference)
**Purpose**: Quick reference for common operations

**Contents**:
- Summary of three modes (train/unlearn/both)
- Key parameters
- Common workflows
- Output interpretation
- Tips for best results

### 4. `README.md` (Project Overview)
**Purpose**: Main project documentation

**Contents**:
- Project overview
- Features
- Quick start
- Installation
- Project structure
- Citation information

### 5. `run.bat` (Windows Quick Start)
**Purpose**: Interactive menu for Windows users

**Features**:
- Menu-driven interface
- Pre-configured common commands
- No need to remember arguments

### 6. `run.sh` (Unix/Linux/Mac Quick Start)
**Purpose**: Interactive menu for Unix-like systems

**Features**:
- Same as run.bat but for Unix shells
- Executable with chmod +x

### 7. `requirements.txt` (Dependencies)
**Purpose**: Easy dependency installation

**Usage**: `pip install -r requirements.txt`

### 8. Configuration Files

#### `configs/quick_test.yaml`
- Fast testing configuration (10 epochs)
- Reduced model size for quick iterations

#### `configs/full_experiment.yaml`
- High-performance configuration (200 epochs)
- Larger models for production use

---

## How to Use

### Method 1: Interactive Menu (Easiest)

**Windows**:
```bash
run.bat
```

**Unix/Linux/Mac**:
```bash
chmod +x run.sh
./run.sh
```

Then select from menu options.

### Method 2: Command Line (Most Flexible)

#### Train a base model and evaluate metrics:
```bash
python train_and_evaluate.py --mode train --dataset movielens-1m --model lightgcn --save_model
```

**What this does**:
- Downloads/loads MovieLens-1M dataset
- Creates LightGCN model
- Pre-trains Influence Encoder (simulates unlearning)
- Evaluates Recall@20 and NDCG@20
- Saves checkpoint

#### Train and perform unlearning:
```bash
python train_and_evaluate.py --mode both --dataset movielens-1m --model lightgcn --save_model
```

**What this does**:
- Everything from above, PLUS:
- Samples edges to unlearn (10% of training set by default)
- Applies Influence Encoder to unlearn
- Evaluates unlearning effectiveness:
  - Score reduction for unlearned edges
  - Comparison with negative samples
- Evaluates model utility (Recall, NDCG on test set)
- Saves results to JSON

#### Load checkpoint and unlearn more:
```bash
python train_and_evaluate.py --mode unlearn --load_checkpoint checkpoints/model.pt
```

### Method 3: Configuration Files (Most Reproducible)

Create a config file or use existing ones:
```bash
python train_and_evaluate.py --config configs/quick_test.yaml --dataset movielens-1m --save_model
```

---

## Key Features

### 1. Flexible Dataset Support
```bash
--dataset movielens-1m    # MovieLens 1M
--dataset gowalla         # Gowalla check-ins
--dataset yelp2018        # Yelp 2018
```
Datasets auto-download on first use!

### 2. Multiple Model Architectures
```bash
--model lightgcn    # Simple and effective
--model sgl         # Self-supervised learning
--model simgcl      # Simplified contrastive learning
```

### 3. Comprehensive Evaluation

**Unlearning Effectiveness**:
- Score reduction percentage
- Comparison with random negative edges
- Ratio metrics (unlearned vs negative)

**Model Utility**:
- Recall@K (coverage)
- NDCG@K (ranking quality)
- Embedding similarity (change magnitude)

### 4. Reproducibility
- Checkpoint saving/loading
- Configuration file support
- Random seed control (`--seed`)
- Results export to JSON with timestamps

### 5. Production Ready
- Train once, unlearn many times (checkpoint reuse)
- Fine-tuning support (`--fine_tune`)
- GPU/CPU selection (`--device`)
- Memory optimization options

---

## Example Workflows

### Quick Test (2-3 minutes)
```bash
python train_and_evaluate.py \
    --mode both \
    --dataset movielens-1m \
    --model lightgcn \
    --num_pretrain_epochs 10 \
    --eval_k 10
```

### Full Experiment (30-60 minutes)
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

### Compare All Models
```bash
for model in lightgcn sgl simgcl; do
    python train_and_evaluate.py --mode both --dataset movielens-1m --model $model --save_model
done
```

### Ablation Study on Unlearning Amount
```bash
for ratio in 0.05 0.10 0.20; do
    python train_and_evaluate.py --mode both --unlearn_test_ratio $ratio --save_model
done
```

---

## Output Files

### Checkpoints (`checkpoints/`)
```
movielens-1m_lightgcn_base_model_20250428_120000.pt
movielens-1m_lightgcn_unlearned_model_20250428_121500.pt
```

**Contains**:
- Model state dict
- Influence Encoder state dict
- All hyperparameters
- Metrics
- Training history

### Results (`results/`)
```
movielens-1m_lightgcn_results_20250428_121500.json
```

**Contains**:
```json
{
  "timestamp": "20250428_121500",
  "args": {...},
  "metrics": {
    "unlearning": {
      "score_reduction_percent": 72.78,
      "avg_original_score": 0.4532,
      "avg_unlearned_score": 0.1234,
      "unlearned_negative_ratio": 1.2503
    },
    "utility": {
      "recall": 0.1189,
      "ndcg": 0.0842
    },
    "embedding_similarity": 0.8934
  }
}
```

---

## Understanding the Metrics

### During Training (Phase 1)
```
Base Model Metrics:
  Recall@20: 0.1234
  NDCG@20: 0.0876
```
This is your baseline - the model's recommendation quality.

### During Unlearning (Phase 2)
```
Score Reduction: 72.78%
```
→ Unlearned edges' scores dropped by ~73%. Good!

```
Unlearned/Negative Score Ratio: 1.25
```
→ Unlearned edges scored 1.25× higher than random negatives.
→ Goal: < 1.0 (perfect unlearning)
→ 1.25 is acceptable (some residual info)

```
Recall@20: 0.1189 (was 0.1234)
NDCG@20: 0.0842 (was 0.0876)
```
→ Utility dropped ~4% - acceptable trade-off!

```
Embedding Similarity: 0.8934
```
→ Model is 89% similar to original - preserved most knowledge

---

## Customization

### Add Your Own Dataset
Edit `data/preprocessor.py`:
```python
@staticmethod
def load_my_dataset(file_path: str):
    # Load your data
    # Return: interactions, num_users, num_items
    pass
```

### Add Your Own Model
Create `models/my_model.py`:
```python
from models.base_gnn import BaseGNN

class MyModel(BaseGNN):
    def forward_with_embeddings(self, embeddings, adj):
        # Your propagation logic
        pass
```

### Add Your Own Metrics
Edit `utils/metrics.py`:
```python
@staticmethod
def my_metric(model, test_edges):
    # Your evaluation logic
    pass
```

---

## Next Steps

1. **Try the quick test**: `python train_and_evaluate.py --mode both --num_pretrain_epochs 10`
2. **Read USAGE.md** for detailed documentation
3. **Experiment with different models/datasets**
4. **Tune hyperparameters** (lambda_u, lambda_p, lambda_c)
5. **Run full experiments** for publication-quality results

---

## Key Advantages

✅ **Easy to use**: Interactive menus or simple commands
✅ **Comprehensive**: Training, unlearning, and evaluation in one tool
✅ **Reproducible**: Checkpoints, configs, and random seeds
✅ **Production-ready**: Load checkpoints and unlearn multiple times
✅ **Well-documented**: Multiple guides for different use cases
✅ **Extensible**: Easy to add datasets, models, and metrics

---

## Questions?

- See `USAGE.md` for detailed documentation
- See `CLI_GUIDE.md` for quick reference
- See `README.md` for project overview
- Check the paper: "Pre-training for Recommendation Unlearning" (SIGIR'25)
