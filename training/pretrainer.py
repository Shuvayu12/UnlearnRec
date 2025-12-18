import torch
import torch.optim as optim
from tqdm import tqdm

class PreTrainer:
    """Pre-training module for Influence Encoder"""
    
    def __init__(self, model, influence_encoder, loss_fn, device='cuda'):
        self.model = model
        self.influence_encoder = influence_encoder
        self.loss_fn = loss_fn
        self.device = device
        
    def pretrain(self, dataset, num_pretrain_epochs=100, unlearn_ratio=0.05):
        """
        Pre-train the influence encoder using simulated unlearning requests
        """
        # Use higher learning rate for pre-training
        optimizer = optim.Adam([
            {'params': self.influence_encoder.H0, 'lr': 0.01},
            {'params': self.influence_encoder.W_eta, 'lr': 0.01}
        ])
        
        history = {
            'total_loss': [],
            'model_loss': [],
            'unlearning_loss': [],
            'preserving_loss': [],
            'contrast_loss': []
        }
        
        # Convert all edges to tensor once (outside loop)
        all_edges_tensor = torch.tensor(dataset.all_edges, dtype=torch.long, device=self.device)
        
        for epoch in tqdm(range(num_pretrain_epochs), desc="Pre-training"):
            # Sample simulated unlearning set
            num_unlearn = int(len(dataset.all_edges) * unlearn_ratio)
            unlearn_indices = torch.randperm(len(dataset.all_edges), device=self.device)[:num_unlearn]
            
            # Create mask for remaining edges (all True except unlearned)
            mask = torch.ones(len(dataset.all_edges), dtype=torch.bool, device=self.device)
            mask[unlearn_indices] = False
            
            unlearn_edges_tensor = all_edges_tensor[unlearn_indices]
            remaining_edges_tensor = all_edges_tensor[mask]
            
            # Construct matrices
            A_delta = self._construct_A_delta(unlearn_edges_tensor, dataset.num_users, dataset.num_items)
            A_r = self._construct_A_r(remaining_edges_tensor, dataset.num_users, dataset.num_items)
            
            # Get original embeddings (ensure they're on the correct device)
            E_original = self.model.get_initial_embeddings().to(self.device)
            
            # Forward pass through influence encoder
            E0_updated = self.influence_encoder(A_delta, E_original)
            
            # Compute H for contrast loss
            with torch.no_grad():
                H = self.influence_encoder.H0.detach()
            
            # Compute loss
            loss_dict = self.loss_fn(
                model=self.model,
                E0_updated=E0_updated,
                A_r=A_r,
                original_embeddings=E_original,
                unlearn_edges=unlearn_edges_tensor,
                remaining_edges=remaining_edges_tensor,
                A_delta=A_delta,
                H=H
            )
            
            # Backward pass with gradient clipping
            optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            
            # Gradient clipping to prevent exploding gradients
            # Record history
            for key in history:
                history[key].append(loss_dict[key].item())
            
            # Print more frequently for short training
            print_interval = max(1, num_pretrain_epochs // 10)
            if epoch % print_interval == 0 or epoch == num_pretrain_epochs - 1:
                print(f"Epoch {epoch}: Total Loss: {loss_dict['total_loss'].item():.4f}, "
                      f"BPR: {loss_dict['model_loss'].item():.4f}, "
                      f"Unlearn: {loss_dict['unlearning_loss'].item():.4f}")
                history[key].append(loss_dict[key].item())
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Total Loss: {loss_dict['total_loss'].item():.4f}")
        
        return history
    
    def _sample_unlearn_edges(self, all_edges, ratio=0.05):
        """Sample edges for simulated unlearning"""
        num_unlearn = int(len(all_edges) * ratio)
        indices = torch.randperm(len(all_edges))[:num_unlearn]
        return [all_edges[i] for i in indices]
    
    def _construct_A_delta(self, edges_tensor, num_users, num_items):
        """Construct influence dependency matrix"""
        num_nodes = num_users + num_items
        A_delta = torch.zeros((num_nodes, num_nodes), device=self.device)
        
        if len(edges_tensor) > 0:
            # edges_tensor is already on device
            users = edges_tensor[:, 0]
            items = edges_tensor[:, 1] + num_users
            
            # Set both directions at once
            A_delta[users, items] = 1
            A_delta[items, users] = 1
            
        return A_delta
    
    def _construct_A_r(self, edges_tensor, num_users, num_items):
        """Construct residual adjacency matrix"""
        num_nodes = num_users + num_items
        A_r = torch.zeros((num_nodes, num_nodes), device=self.device)
        
        if len(edges_tensor) > 0:
            # edges_tensor is already on device
            users = edges_tensor[:, 0]
            items = edges_tensor[:, 1] + num_users
            
            # Set both directions at once
            A_r[users, items] = 1
            A_r[items, users] = 1
            
        return A_r