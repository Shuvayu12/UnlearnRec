"""
Command-line interface for training and evaluating UnlearnRec
"""
import sys
import os

# Add the current directory to Python path for module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import torch
import yaml
import numpy as np
from datetime import datetime

from data.preprocessor import DataPreprocessor
from data.dataset import RecommendationDataset
from models.lightgcn import LightGCN
from models.sgl import SGL
from models.simgcl import SimGCL
from core.influence_encoder import InfluenceEncoder
from core.loss_function import UnlearnRecLoss
from core.unlearning_manager import UnlearningManager
from training.base_trainer import BaseModelTrainer
from training.pretrainer import PreTrainer
from utils.metrics import UnlearningMetrics


def parse_args():
    parser = argparse.ArgumentParser(description='UnlearnRec: Training and Unlearning for Recommendation Systems')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='both', choices=['train', 'unlearn', 'both'],
                        help='Mode: train (base model only), unlearn (unlearning only), both (train then unlearn)')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='movielens-1m',
                        choices=['movielens-1m', 'gowalla', 'yelp2018'],
                        help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory for dataset storage')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Ratio of test set')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='lightgcn',
                        choices=['lightgcn', 'sgl', 'simgcl'],
                        help='Base GNN model to use')
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='Embedding dimension')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--num_layers_ie', type=int, default=3,
                        help='Number of layers in Influence Encoder')
    parser.add_argument('--num_layers_mlp', type=int, default=2,
                        help='Number of MLP layers in Influence Encoder')
    
    # Training parameters
    parser.add_argument('--num_base_epochs', type=int, default=50,
                        help='Number of base model training epochs')
    parser.add_argument('--num_pretrain_epochs', type=int, default=100,
                        help='Number of pre-training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size')
    parser.add_argument('--reg_weight', type=float, default=1e-4,
                        help='L2 regularization weight')
    
    # Unlearning parameters
    parser.add_argument('--unlearn_ratio', type=float, default=0.05,
                        help='Ratio of edges to unlearn during pre-training')
    parser.add_argument('--lambda_u', type=float, default=1.0,
                        help='Weight for unlearning loss')
    parser.add_argument('--lambda_p', type=float, default=1.0,
                        help='Weight for preserving loss')
    parser.add_argument('--lambda_c', type=float, default=0.01,
                        help='Weight for contrast loss')
    parser.add_argument('--fine_tune', action='store_true',
                        help='Enable fine-tuning after unlearning')
    parser.add_argument('--fine_tune_epochs', type=int, default=3,
                        help='Number of fine-tuning epochs')
    parser.add_argument('--unlearn_test_ratio', type=float, default=0.1,
                        help='Ratio of edges to unlearn in test phase')
    
    # Evaluation parameters
    parser.add_argument('--eval_k', type=int, default=20,
                        help='K for Recall@K and NDCG@K metrics')
    
    # I/O parameters
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (overrides command-line args)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--save_model', action='store_true',
                        help='Save trained models')
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help='Path to checkpoint to load')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use')
    
    return parser.parse_args()


def load_config_from_file(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def merge_config(args, config_dict):
    """Merge command-line args with config file (args take precedence)"""
    for key, value in config_dict.items():
        if not hasattr(args, key) or getattr(args, key) == argparse.ArgumentParser().parse_args([]).__dict__.get(key):
            setattr(args, key, value)
    return args


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg):
    """Get PyTorch device"""
    if device_arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_arg)


def create_model(args, num_users, num_items):
    """Create GNN model based on arguments"""
    if args.model == 'lightgcn':
        return LightGCN(num_users, num_items, args.embedding_dim, args.num_layers)
    elif args.model == 'sgl':
        return SGL(num_users, num_items, args.embedding_dim, args.num_layers)
    elif args.model == 'simgcl':
        return SimGCL(num_users, num_items, args.embedding_dim, args.num_layers)
    else:
        raise ValueError(f"Unknown model: {args.model}")


def train_base_model(args, dataset, device):
    """Train the base recommendation model"""
    print("\n" + "="*80)
    print("PHASE 1: Training Base Recommendation Model")
    print("="*80)
    
    # Create model
    model = create_model(args, dataset.num_users, dataset.num_items).to(device)
    print(f"Model: {args.model}")
    print(f"Users: {dataset.num_users}, Items: {dataset.num_items}")
    print(f"Total interactions: {len(dataset.all_edges)}")
    
    # Split dataset
    train_edges, test_edges = dataset.split_edges(test_ratio=args.test_ratio)
    print(f"Train edges: {len(train_edges)}, Test edges: {len(test_edges)}")
    
    # Convert train edges to tensor
    train_edges_tensor = torch.tensor(train_edges, dtype=torch.long, device=device)
    
    # STEP 1: Train base model
    print("\n" + "-"*80)
    print("STEP 1: Training Base Model")
    print("-"*80)
    base_trainer = BaseModelTrainer(model, device, args.learning_rate, args.reg_weight)
    base_history = base_trainer.train(
        train_edges_tensor,
        dataset.num_users,
        dataset.num_items,
        num_epochs=args.num_base_epochs,
        batch_size=args.batch_size
    )
    
    # Evaluate after base training
    print("\n" + "-"*80)
    print("Evaluating Base Model Performance")
    print("-"*80)
    
    train_dataset = RecommendationDataset(train_edges, dataset.num_users, dataset.num_items)
    adj = train_dataset.get_adjacency_matrix().to(device)
    embeddings = model(adj)
    
    metrics = evaluate_model(model, embeddings, test_edges, dataset.num_users, args.eval_k, device)
    
    print(f"\nBase Model Metrics:")
    print(f"  Recall@{args.eval_k}: {metrics['recall']:.4f}")
    print(f"  NDCG@{args.eval_k}: {metrics['ndcg']:.4f}")
    
    # STEP 2: Pre-train influence encoder
    print("\n" + "-"*80)
    print("STEP 2: Pre-training Influence Encoder")
    print("-"*80)
    
    # Initialize influence encoder
    num_nodes = dataset.num_users + dataset.num_items
    influence_encoder = InfluenceEncoder(
        num_nodes=num_nodes,
        embedding_dim=args.embedding_dim,
        num_layers_ie=args.num_layers_ie,
        num_layers_mlp=args.num_layers_mlp
    ).to(device)
    
    # Initialize loss function
    loss_fn = UnlearnRecLoss(
        lambda_u=args.lambda_u,
        lambda_p=args.lambda_p,
        lambda_c=args.lambda_c
    )
    
    # Pre-train the influence encoder
    pretrainer = PreTrainer(model, influence_encoder, loss_fn, device)
    pretrain_history = pretrainer.pretrain(
        train_dataset,
        num_pretrain_epochs=args.num_pretrain_epochs,
        unlearn_ratio=args.unlearn_ratio
    )
    
    # Freeze pre-trained parameters
    influence_encoder.freeze_pretrained_params()
    
    # Save checkpoint if requested
    if args.save_model:
        save_checkpoint(args, model, influence_encoder, pretrain_history, metrics, 'base_model')
    
    return model, influence_encoder, train_edges, test_edges, metrics


def perform_unlearning(args, model, influence_encoder, train_edges, test_edges, dataset, device):
    """Perform unlearning and evaluation"""
    print("\n" + "="*80)
    print("PHASE 2: Unlearning and Evaluation")
    print("="*80)
    
    # Sample edges to unlearn from training set
    num_unlearn = int(len(train_edges) * args.unlearn_test_ratio)
    indices = torch.randperm(len(train_edges))[:num_unlearn]
    unlearn_edges = [train_edges[i] for i in indices]
    remaining_edges = [train_edges[i] for i in range(len(train_edges)) if i not in indices]
    
    print(f"Unlearning {num_unlearn} edges ({args.unlearn_test_ratio*100:.1f}% of training set)")
    print(f"Remaining edges: {len(remaining_edges)}")
    
    # Get original embeddings and scores for unlearned edges
    adj_original = dataset.get_adjacency_matrix(train_edges).to(device)
    embeddings_original = model(adj_original)
    
    original_scores = []
    with torch.no_grad():
        for u, v in unlearn_edges:
            user_emb = embeddings_original[u]
            item_emb = embeddings_original[v + dataset.num_users]
            score = torch.dot(user_emb, item_emb).item()
            original_scores.append((u, v, score))
    
    print(f"Average original score for unlearn edges: {np.mean([s for _, _, s in original_scores]):.4f}")
    
    # Initialize unlearning manager
    unlearning_manager = UnlearningManager(model, influence_encoder, device)
    
    # Process unlearning request
    print("\nProcessing unlearning request...")
    result = unlearning_manager.process_unlearning_request(
        unlearn_edges=unlearn_edges,
        all_edges=train_edges,
        num_users=dataset.num_users,
        num_items=dataset.num_items,
        fine_tune=args.fine_tune,
        fine_tune_epochs=args.fine_tune_epochs
    )
    
    unlearned_model = result['unlearned_model']
    
    # Compute unlearned scores
    adj_remaining = dataset.get_adjacency_matrix(remaining_edges).to(device)
    embeddings_unlearned = unlearned_model(adj_remaining)
    
    unlearned_scores = []
    with torch.no_grad():
        for u, v in unlearn_edges:
            user_emb = embeddings_unlearned[u]
            item_emb = embeddings_unlearned[v + dataset.num_users]
            score = torch.dot(user_emb, item_emb).item()
            unlearned_scores.append(score)
    
    print(f"Average unlearned score for unlearn edges: {np.mean(unlearned_scores):.4f}")
    
    # Sample negative edges for comparison
    negative_edges = DataPreprocessor.sample_negative_edges(
        train_edges, dataset.num_users, dataset.num_items, num_negative=len(unlearn_edges)
    )
    
    negative_scores = []
    with torch.no_grad():
        for u, v in negative_edges:
            user_emb = embeddings_unlearned[u]
            item_emb = embeddings_unlearned[v + dataset.num_users]
            score = torch.dot(user_emb, item_emb).item()
            negative_scores.append(score)
    
    print(f"Average score for negative edges: {np.mean(negative_scores):.4f}")
    
    # Evaluate unlearning effectiveness
    print("\n" + "-"*80)
    print("Unlearning Effectiveness Metrics")
    print("-"*80)
    
    # Score reduction
    avg_original = np.mean([s for _, _, s in original_scores])
    avg_unlearned = np.mean(unlearned_scores)
    score_reduction = (avg_original - avg_unlearned) / avg_original * 100
    print(f"Score Reduction: {score_reduction:.2f}%")
    
    # Comparison with negative edges
    avg_negative = np.mean(negative_scores)
    ratio_to_negative = avg_unlearned / avg_negative if avg_negative != 0 else float('inf')
    print(f"Unlearned/Negative Score Ratio: {ratio_to_negative:.4f}")
    print(f"  (Lower is better, <1.0 means unlearned edges scored below random negatives)")
    
    # Evaluate model utility on test set
    print("\n" + "-"*80)
    print("Model Utility Metrics (on Test Set)")
    print("-"*80)
    
    utility_metrics = evaluate_model(unlearned_model, embeddings_unlearned, test_edges, 
                                    dataset.num_users, args.eval_k, device)
    
    print(f"Recall@{args.eval_k}: {utility_metrics['recall']:.4f}")
    print(f"NDCG@{args.eval_k}: {utility_metrics['ndcg']:.4f}")
    
    # Compile all metrics
    all_metrics = {
        'unlearning': {
            'score_reduction_percent': score_reduction,
            'avg_original_score': avg_original,
            'avg_unlearned_score': avg_unlearned,
            'avg_negative_score': avg_negative,
            'unlearned_negative_ratio': ratio_to_negative,
            'num_unlearned_edges': len(unlearn_edges)
        },
        'utility': utility_metrics
    }
    
    # Embedding similarity
    emb_similarity = torch.nn.functional.cosine_similarity(
        embeddings_original.flatten().unsqueeze(0),
        embeddings_unlearned.flatten().unsqueeze(0)
    ).item()
    all_metrics['embedding_similarity'] = emb_similarity
    print(f"\nEmbedding Similarity (Original vs Unlearned): {emb_similarity:.4f}")
    
    # Save results
    if args.save_model:
        save_checkpoint(args, unlearned_model, influence_encoder, None, all_metrics, 'unlearned_model')
    
    save_results(args, all_metrics)
    
    return unlearned_model, all_metrics


def evaluate_model(model, embeddings, test_edges, num_users, k, device):
    """Evaluate model on test edges"""
    # Split embeddings into user and item
    user_embeddings = embeddings[:num_users]
    item_embeddings = embeddings[num_users:]
    
    # Compute all predictions (keep on GPU, no grad needed)
    with torch.no_grad():
        predictions = torch.matmul(user_embeddings, item_embeddings.t())
    
    # Build test set per user
    user_test_items = {}
    for u, v in test_edges:
        if u not in user_test_items:
            user_test_items[u] = []
        user_test_items[u].append(v)
    
    # Compute Recall@K and NDCG@K (vectorized where possible)
    recalls = []
    ndcgs = []
    
    # Move to CPU only once for sorting (detach if needed)
    predictions_cpu = predictions.detach().cpu()
    
    for user, test_items in user_test_items.items():
        if len(test_items) == 0:
            continue
        
        # Get predictions for this user (already on CPU)
        user_scores = predictions_cpu[user].numpy()
        
        # Get top-K items
        top_k_items = user_scores.argsort()[-k:][::-1].tolist()
        
        # Recall@K
        hits = len(set(test_items) & set(top_k_items))
        recall = hits / min(len(test_items), k)
        recalls.append(recall)
        
        # NDCG@K
        dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(top_k_items) if item in test_items)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(test_items), k)))
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)
    
    return {
        'recall': np.mean(recalls) if recalls else 0.0,
        'ndcg': np.mean(ndcgs) if ndcgs else 0.0
    }


def save_checkpoint(args, model, influence_encoder, history, metrics, suffix):
    """Save model checkpoint"""
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(
        args.checkpoint_dir,
        f"{args.dataset}_{args.model}_{suffix}_{timestamp}.pt"
    )
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'influence_encoder_state_dict': influence_encoder.state_dict(),
        'args': vars(args),
        'metrics': metrics,
        'history': history
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"\nCheckpoint saved to: {checkpoint_path}")


def save_results(args, metrics):
    """Save evaluation results to JSON"""
    os.makedirs(args.results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(
        args.results_dir,
        f"{args.dataset}_{args.model}_results_{timestamp}.json"
    )
    
    results = {
        'timestamp': timestamp,
        'args': vars(args),
        'metrics': metrics
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")


def main():
    # Parse arguments
    args = parse_args()
    
    # Load config file if specified
    if args.config:
        config = load_config_from_file(args.config)
        args = merge_config(args, config)
    
    # Set seed
    set_seed(args.seed)
    
    # Get device
    device = get_device(args.device)
    print(f"\n{'='*80}")
    print("Device Information")
    print(f"{'='*80}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"Using device: {device}")
    
    # Verify device is actually CUDA if requested
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("\n⚠️  WARNING: CUDA requested but not available! Falling back to CPU.")
        print("Make sure GPU is enabled in Kaggle notebook settings.")
    
    # Load dataset
    print("\n" + "="*80)
    print("Loading Dataset")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Data directory: {args.data_dir}")
    
    dataset = DataPreprocessor.build_recommendation_dataset(args.dataset, args.data_dir)
    print(f"Loaded {len(dataset)} interactions")
    print(f"Users: {dataset.num_users}, Items: {dataset.num_items}")
    
    # Execute based on mode
    if args.mode in ['train', 'both']:
        model, influence_encoder, train_edges, test_edges, base_metrics = train_base_model(
            args, dataset, device
        )
        
        if args.mode == 'train':
            print("\n" + "="*80)
            print("Training Complete!")
            print("="*80)
            return
    
    if args.mode in ['unlearn', 'both']:
        if args.mode == 'unlearn':
            # Load checkpoint
            if not args.load_checkpoint:
                raise ValueError("Must specify --load_checkpoint for unlearn-only mode")
            
            print(f"\nLoading checkpoint from: {args.load_checkpoint}")
            checkpoint = torch.load(args.load_checkpoint, map_location=device)
            
            # Recreate model and influence encoder
            dataset_full = DataPreprocessor.build_recommendation_dataset(args.dataset, args.data_dir)
            model = create_model(args, dataset_full.num_users, dataset_full.num_items).to(device)
            
            num_nodes = dataset_full.num_users + dataset_full.num_items
            influence_encoder = InfluenceEncoder(
                num_nodes=num_nodes,
                embedding_dim=args.embedding_dim,
                num_layers_ie=args.num_layers_ie,
                num_layers_mlp=args.num_layers_mlp
            ).to(device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            influence_encoder.load_state_dict(checkpoint['influence_encoder_state_dict'])
            
            # Split dataset
            train_edges, test_edges = dataset_full.split_edges(test_ratio=args.test_ratio)
            dataset = dataset_full
        
        # Perform unlearning
        unlearned_model, unlearning_metrics = perform_unlearning(
            args, model, influence_encoder, train_edges, test_edges, dataset, device
        )
    
    print("\n" + "="*80)
    print("All Operations Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
