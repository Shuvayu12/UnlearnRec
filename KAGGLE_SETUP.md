# Running UnlearnRec on Kaggle

This guide explains how to run UnlearnRec on Kaggle notebooks after cloning from GitHub.

## Method 1: Clone from GitHub in Kaggle Notebook

### Step 1: Create a New Kaggle Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Click "New Notebook"
3. Enable GPU (Settings → Accelerator → GPU T4 x2 or P100)

### Step 2: Clone the Repository

In the first cell:

```python
# Clone the repository
!git clone https://github.com/YOUR_USERNAME/UnlearnRec.git
%cd UnlearnRec
```

### Step 3: Install Dependencies

```python
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q -r requirements.txt
```

### Step 4: Verify Installation

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

### Step 5: Run Quick Test

```python
!python train_and_evaluate.py \
    --mode both \
    --dataset movielens-1m \
    --model lightgcn \
    --num_pretrain_epochs 10 \
    --eval_k 10 \
    --device cuda
```

---

## Method 2: Upload Files Directly to Kaggle

### Step 1: Download Repository

1. Download the repository as ZIP from GitHub
2. Extract locally

### Step 2: Create Kaggle Dataset

1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets)
2. Click "New Dataset"
3. Upload all files (excluding .git, data/, checkpoints/, results/)
4. Name it "unlearnrec-code"

### Step 3: Add Dataset to Notebook

1. Create new notebook
2. Add your dataset under "Input"
3. Copy files to working directory:

```python
import shutil
import os

# Copy code files
source = '/kaggle/input/unlearnrec-code'
!cp -r {source}/* .

# Create necessary directories
!mkdir -p data checkpoints results
```

### Step 4: Install and Run

```python
!pip install -q -r requirements.txt

# Run training
!python train_and_evaluate.py --mode both --num_pretrain_epochs 10
```

---

## Kaggle-Specific Configurations

### Memory Optimization for Kaggle

Kaggle notebooks have memory limits. For larger datasets, use these settings:

```python
!python train_and_evaluate.py \
    --mode both \
    --dataset movielens-1m \
    --model lightgcn \
    --embedding_dim 32 \
    --num_layers 2 \
    --batch_size 256 \
    --num_pretrain_epochs 50 \
    --device cuda
```

### Time Limits

Kaggle notebooks have execution time limits:
- Free tier: 9 hours (GPU), 12 hours (CPU)
- Verified: 12 hours (GPU)

For quick experiments within time limits:

```python
!python train_and_evaluate.py \
    --mode both \
    --num_pretrain_epochs 50 \
    --save_model
```

### Saving Results

Kaggle notebooks don't persist files after session. To save results:

#### Option A: Download Results

```python
from IPython.display import FileLink
import os

# After training, download checkpoints
for file in os.listdir('checkpoints'):
    display(FileLink(f'checkpoints/{file}'))

# Download results
for file in os.listdir('results'):
    display(FileLink(f'results/{file}'))
```

#### Option B: Save to Kaggle Datasets

```python
# After training, create output dataset
# Kaggle will automatically save /kaggle/working/ as output
!cp -r checkpoints /kaggle/working/
!cp -r results /kaggle/working/
```

Then commit the notebook, and outputs will be saved as a dataset.

---

## Complete Kaggle Notebook Example

Here's a complete example notebook structure:

### Cell 1: Setup

```python
# Clone repository
!git clone https://github.com/YOUR_USERNAME/UnlearnRec.git
%cd UnlearnRec

# Install dependencies
!pip install -q torch torchvision torchaudio
!pip install -q -r requirements.txt

# Verify GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

### Cell 2: Quick Test

```python
# Quick test (2-3 minutes)
!python train_and_evaluate.py \
    --mode both \
    --dataset movielens-1m \
    --model lightgcn \
    --num_pretrain_epochs 5 \
    --eval_k 10 \
    --device cuda
```

### Cell 3: Full Training

```python
# Full training (~30-60 minutes)
!python train_and_evaluate.py \
    --mode both \
    --dataset movielens-1m \
    --model lightgcn \
    --num_pretrain_epochs 100 \
    --fine_tune \
    --fine_tune_epochs 5 \
    --eval_k 20 \
    --save_model \
    --device cuda
```

### Cell 4: View Results

```python
import json

# Read latest results
import os
results_files = sorted(os.listdir('results'))
if results_files:
    latest = results_files[-1]
    with open(f'results/{latest}', 'r') as f:
        results = json.load(f)
    
    print("=== Unlearning Effectiveness ===")
    print(f"Score Reduction: {results['metrics']['unlearning']['score_reduction_percent']:.2f}%")
    print(f"Unlearned/Negative Ratio: {results['metrics']['unlearning']['unlearned_negative_ratio']:.4f}")
    
    print("\n=== Model Utility ===")
    print(f"Recall@20: {results['metrics']['utility']['recall']:.4f}")
    print(f"NDCG@20: {results['metrics']['utility']['ndcg']:.4f}")
    
    print(f"\n=== Embedding Similarity ===")
    print(f"{results['metrics']['embedding_similarity']:.4f}")
```

### Cell 5: Download Results

```python
from IPython.display import FileLink

print("Download checkpoints:")
for file in os.listdir('checkpoints'):
    display(FileLink(f'checkpoints/{file}'))

print("\nDownload results:")
for file in os.listdir('results'):
    display(FileLink(f'results/{file}'))
```

---

## Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce model size

```python
!python train_and_evaluate.py \
    --embedding_dim 32 \
    --batch_size 128 \
    --num_layers 2
```

### Issue: Dataset Download Fails

**Solution**: Use Kaggle datasets

1. Add MovieLens-1M dataset from Kaggle
2. Modify data path:

```python
!python train_and_evaluate.py \
    --data_dir /kaggle/input/movielens-1m \
    --mode both
```

Or manually download:

```python
!wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
!unzip ml-1m.zip -d data/
```

### Issue: Time Limit Exceeded

**Solution**: Reduce epochs

```python
!python train_and_evaluate.py \
    --num_pretrain_epochs 30 \
    --mode both
```

### Issue: Import Errors

**Solution**: Ensure you're in the correct directory

```python
%cd /kaggle/working/UnlearnRec
!pwd
!ls
```

---

## Best Practices for Kaggle

1. **Enable GPU**: Always use GPU for faster training
2. **Start Small**: Test with `--num_pretrain_epochs 10` first
3. **Save Early**: Use `--save_model` to save checkpoints
4. **Monitor Progress**: Check output regularly
5. **Download Results**: Save important files before session ends
6. **Use Configs**: Create and use config files for reproducibility

---

## Example Configurations for Kaggle

### Quick Test Config (5 minutes)

```yaml
# save as kaggle_quick.yaml
embedding_dim: 32
num_layers: 2
num_pretrain_epochs: 10
eval_k: 10
test_ratio: 0.2
```

Run with:
```python
!python train_and_evaluate.py --config kaggle_quick.yaml --mode both
```

### Full Experiment Config (1 hour)

```yaml
# save as kaggle_full.yaml
embedding_dim: 64
num_layers: 3
num_pretrain_epochs: 100
fine_tune: true
fine_tune_epochs: 5
eval_k: 20
test_ratio: 0.2
```

Run with:
```python
!python train_and_evaluate.py --config kaggle_full.yaml --mode both --save_model
```

---

## Template Notebook

Here's a ready-to-use Kaggle notebook template:

```python
# ==================================================
# UnlearnRec on Kaggle - Complete Template
# ==================================================

# 1. Setup
!git clone https://github.com/YOUR_USERNAME/UnlearnRec.git
%cd UnlearnRec
!pip install -q -r requirements.txt

# 2. Verify Environment
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

# 3. Quick Test (3 minutes)
!python train_and_evaluate.py --mode both --num_pretrain_epochs 5

# 4. Full Training (choose one based on time available)

# Option A: Fast (30 min)
!python train_and_evaluate.py --mode both --num_pretrain_epochs 50 --save_model

# Option B: Complete (60 min)
!python train_and_evaluate.py --mode both --num_pretrain_epochs 100 --fine_tune --save_model

# 5. View Results
import json, os
results_file = sorted(os.listdir('results'))[-1]
with open(f'results/{results_file}') as f:
    print(json.dumps(json.load(f)['metrics'], indent=2))

# 6. Download (optional)
from IPython.display import FileLink
for f in os.listdir('checkpoints'): display(FileLink(f'checkpoints/{f}'))
for f in os.listdir('results'): display(FileLink(f'results/{f}'))
```

---

## Summary

✅ Clone from GitHub or upload as dataset  
✅ Install dependencies with pip  
✅ Enable GPU for faster training  
✅ Start with quick test (10 epochs)  
✅ Save and download results  
✅ Use configurations for reproducibility  

**Estimated Times on Kaggle GPU:**
- Quick test (10 epochs): 3-5 minutes
- Medium (50 epochs): 15-30 minutes  
- Full (100 epochs): 30-60 minutes

For questions, see the main [README.md](README.md) and [USAGE.md](USAGE.md).
