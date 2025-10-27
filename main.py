import torch
import yaml
from data.dataset import RecommendationDataset
from models.lightgcn import LightGCN
from core.influence_encoder import InfluenceEncoder
from core.loss_function import UnlearnRecLoss
from core.unlearning_manager import UnlearningManager
from training.pretrainer import PreTrainer

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Load configuration
    config = load_config('configs/base.yaml')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset (example)
    # In practice, you would load from files like Movielens, Gowalla, etc.
    user_item_interactions = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 1)]  # Example data
    num_users = 3
    num_items = 3
    
    dataset = RecommendationDataset(user_item_interactions, num_users, num_items)
    
    # Initialize model
    model = LightGCN(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=config['embedding_dim'],
        num_layers=config['num_layers']
    ).to(device)
    
    # Initialize influence encoder
    num_nodes = num_users + num_items
    influence_encoder = InfluenceEncoder(
        num_nodes=num_nodes,
        embedding_dim=config['embedding_dim'],
        num_layers_ie=config['num_layers_ie'],
        num_layers_mlp=config['num_layers_mlp']
    ).to(device)
    
    # Initialize loss function
    loss_fn = UnlearnRecLoss(
        lambda_u=config['lambda_u'],
        lambda_p=config['lambda_p'],
        lambda_c=config['lambda_c']
    )
    
    # Pre-training phase
    print("Starting pre-training...")
    pretrainer = PreTrainer(model, influence_encoder, loss_fn, device)
    pretrain_history = pretrainer.pretrain(
        dataset, 
        num_pretrain_epochs=config['num_pretrain_epochs'],
        unlearn_ratio=config['unlearn_ratio']
    )
    
    # Freeze pre-trained parameters
    influence_encoder.freeze_pretrained_params()
    
    # Initialize unlearning manager
    unlearning_manager = UnlearningManager(model, influence_encoder, device)
    
    # Example unlearning request
    unlearn_edges = [(0, 1)]  # User 0 wants to unlearn interaction with item 1
    
    print("Processing unlearning request...")
    result = unlearning_manager.process_unlearning_request(
        unlearn_edges=unlearn_edges,
        all_edges=dataset.all_edges,
        num_users=num_users,
        num_items=num_items,
        fine_tune=config['fine_tune'],
        fine_tune_epochs=config['fine_tune_epochs']
    )
    
    print("Unlearning completed!")
    print(f"Original model embeddings shape: {model.get_initial_embeddings().shape}")
    print(f"Unlearned model embeddings shape: {result['updated_embeddings'].shape}")
    
    return result

if __name__ == "__main__":
    result = main()