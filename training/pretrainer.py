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
        optimizer = optim.Adam([
            {'params': self.influence_encoder.H0},
            {'params': self.influence_encoder.W_eta}
        ], lr=0.001)
        
        history = {
            'total_loss': [],
            'model_loss': [],
            'unlearning_loss': [],
            'preserving_loss': [],
            'contrast_loss': []
        }
        
        for epoch in tqdm(range(num_pretrain_epochs), desc="Pre-training"):
            # Sample simulated unlearning set
            unlearn_edges = self._sample_unlearn_edges(
                dataset.all_edges, 
                ratio=unlearn_ratio
            )
            unlearn_edges_set = {tuple(edge) for edge in unlearn_edges}  # Set for faster lookups
            remaining_edges = [e for e in dataset.all_edges if tuple(e) not in unlearn_edges_set]
            
            # Construct matrices
            A_delta = self._construct_A_delta(unlearn_edges, dataset.num_users, dataset.num_items)
            A_r = self._construct_A_r(remaining_edges, dataset.num_users, dataset.num_items)
            
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
                unlearn_edges=unlearn_edges,
                remaining_edges=remaining_edges,
                A_delta=A_delta,
                H=H
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            optimizer.step()
            
            # Record history
            for key in history:
                history[key].append(loss_dict[key].item())
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Total Loss: {loss_dict['total_loss'].item():.4f}")
        
        return history
    
    def _sample_unlearn_edges(self, all_edges, ratio=0.05):
        """Sample edges for simulated unlearning"""
        num_unlearn = int(len(all_edges) * ratio)
        indices = torch.randperm(len(all_edges))[:num_unlearn]
        return [all_edges[i] for i in indices]
    
    def _construct_A_delta(self, unlearn_edges, num_users, num_items):
        """Construct influence dependency matrix"""
        num_nodes = num_users + num_items
        A_delta = torch.zeros((num_nodes, num_nodes), device=self.device)
        
        if len(unlearn_edges) > 0:
            # Vectorized edge construction
            edges_tensor = torch.tensor(unlearn_edges, device=self.device)
            users = edges_tensor[:, 0]
            items = edges_tensor[:, 1] + num_users
            
            # Set both directions at once
            A_delta[users, items] = 1
            A_delta[items, users] = 1
            
        return A_delta
    
    def _construct_A_r(self, remaining_edges, num_users, num_items):
        """Construct residual adjacency matrix"""
        num_nodes = num_users + num_items
        A_r = torch.zeros((num_nodes, num_nodes), device=self.device)
        
        if len(remaining_edges) > 0:
            # Vectorized edge construction
            edges_tensor = torch.tensor(remaining_edges, device=self.device)
            users = edges_tensor[:, 0]
            items = edges_tensor[:, 1] + num_users
            
            # Set both directions at once
            A_r[users, items] = 1
            A_r[items, users] = 1
            
        return A_r