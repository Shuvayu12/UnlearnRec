import torch
import torch.nn as nn
import torch.nn.functional as F

class UnlearnRecLoss(nn.Module):
    """Multi-task loss function for UnlearnRec"""
    
    def __init__(self, lambda_u=1.0, lambda_p=1.0, lambda_c=0.01, temperature=1.0):
        super(UnlearnRecLoss, self).__init__()
        self.lambda_u = lambda_u
        self.lambda_p = lambda_p
        self.lambda_c = lambda_c
        self.temperature = temperature
    
    def model_loss(self, model, E0_updated, A_r, remaining_edges):
        """
        Compute model-specific loss including SSL and BPR losses
        """
        # Forward pass with updated embeddings
        embeddings_final = model.forward_with_embeddings(E0_updated, A_r)
        
        # Compute predictions from embeddings
        predictions = model.predict(E0_updated, A_r)
        
        # BPR loss for remaining edges
        bpr_loss = model.compute_bpr_loss(predictions, remaining_edges)
        
        # SSL loss if applicable
        ssl_loss = 0
        if hasattr(model, 'compute_ssl_loss'):
            ssl_loss = model.compute_ssl_loss(embeddings_final)
        
        return bpr_loss + ssl_loss, embeddings_final
    
    def unlearning_loss(self, predictions, unlearn_edges):
        """
        Enforce decrease in predicted scores for unlearned edges (Eq. 20)
        """
        if len(unlearn_edges) == 0:
            return torch.tensor(0.0, device=predictions.device)
        
        # Handle both tensor and list inputs
        if isinstance(unlearn_edges, torch.Tensor):
            users = unlearn_edges[:, 0]
            items = unlearn_edges[:, 1]
        else:
            users = torch.tensor([u for u, _ in unlearn_edges], device=predictions.device)
            items = torch.tensor([v for _, v in unlearn_edges], device=predictions.device)
        
        scores = predictions[users, items]
        # Add epsilon for numerical stability
        loss = -torch.log(torch.sigmoid(-scores) + 1e-10).mean()
        
        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=predictions.device)
        
        return loss
    
    def preserving_loss(self, original_embeddings, updated_embeddings, remaining_edges, tau=1.0):
        """
        Preserve embedding distribution of remaining positive pairs (Eq. 21-22)
        Simplified to use MSE between embeddings for computational efficiency
        """
        if len(remaining_edges) == 0:
            return torch.tensor(0.0, device=original_embeddings.device)
        
        # Sample a subset if too large (for computational efficiency)
        max_edges = 5000
        if isinstance(remaining_edges, torch.Tensor):
            num_edges = len(remaining_edges)
            if num_edges > max_edges:
                indices = torch.randperm(num_edges, device=remaining_edges.device)[:max_edges]
                edges_sample = remaining_edges[indices]
            else:
                edges_sample = remaining_edges
            users = edges_sample[:, 0]
            items = edges_sample[:, 1]
        else:
            num_edges = len(remaining_edges)
            if num_edges > max_edges:
                import random
                edges_sample = random.sample(remaining_edges, max_edges)
            else:
                edges_sample = remaining_edges
            users = torch.tensor([u for u, _ in edges_sample], device=original_embeddings.device)
            items = torch.tensor([v for _, v in edges_sample], device=original_embeddings.device)
        
        # Compute similarity preservation between original and updated
        orig_sim = (original_embeddings[users] * original_embeddings[items]).sum(dim=1)
        updated_sim = (updated_embeddings[users] * updated_embeddings[items]).sum(dim=1)
        
        return F.mse_loss(updated_sim, orig_sim)
    
    def contrast_loss(self, H, A_delta, dropout_rate=0.1):
        """
        Contrastive loss for influence generalization (Eq. 23-24)
        """
        # Apply dropout to A_delta
        mask = torch.rand_like(A_delta) > dropout_rate
        A_delta_prime = A_delta * mask.float()
        
        # Compute degree matrix for dropped version (avoid dense diagonal)
        degree_prime = torch.sum(A_delta_prime, dim=1)
        degree_inv_sqrt = torch.pow(degree_prime + 1e-10, -0.5)
        degree_inv_sqrt = torch.clamp(degree_inv_sqrt, min=0.0, max=1e10)
        degree_inv_sqrt[torch.isnan(degree_inv_sqrt)] = 0.0
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        
        # Normalized adjacency using element-wise multiplication (not diagonal matrix)
        A_norm_prime = A_delta_prime * degree_inv_sqrt.view(-1, 1) * degree_inv_sqrt.view(1, -1)
        
        # Compute H' with dropped adjacency
        H_prime = A_norm_prime @ H
        
        # Contrastive alignment (simplified)
        cos_sim = F.cosine_similarity(H, H_prime, dim=1)
        # Clamp cosine similarity to avoid extreme values
        cos_sim = torch.clamp(cos_sim, min=-1.0, max=1.0)
        loss = -torch.log(torch.sigmoid(cos_sim / self.temperature) + 1e-10).mean()
        
        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=H.device)
        
        return loss
    
    def forward(self, model, E0_updated, A_r, original_embeddings, 
                unlearn_edges, remaining_edges, A_delta, H):
        """
        Compute total loss (Eq. 25)
        """
        # Model loss (computed on remaining edges, not unlearn edges)
        L_m, updated_embeddings_final = self.model_loss(model, E0_updated, A_r, remaining_edges)
        
        # Unlearning loss
        predictions = model.predict(E0_updated, A_r)
        L_u = self.unlearning_loss(predictions, unlearn_edges)
        
        # Preserving loss
        L_p = self.preserving_loss(original_embeddings, updated_embeddings_final, remaining_edges)
        
        # Contrast loss
        L_c = self.contrast_loss(H, A_delta)
        
        total_loss = L_m + self.lambda_u * L_u + self.lambda_p * L_p + self.lambda_c * L_c
        
        return {
            'total_loss': total_loss,
            'model_loss': L_m,
            'unlearning_loss': L_u,
            'preserving_loss': L_p,
            'contrast_loss': L_c
        }