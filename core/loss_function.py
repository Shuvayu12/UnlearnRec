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
        loss = 0
        for u, v in unlearn_edges:
            score = predictions[u, v]
            # We want scores to be negative (low probability)
            loss += -torch.log(torch.sigmoid(-score))
        
        return loss / len(unlearn_edges) if len(unlearn_edges) > 0 else torch.tensor(0.0)
    
    def preserving_loss(self, original_embeddings, updated_embeddings, remaining_edges, tau=1.0):
        """
        Preserve embedding distribution of remaining positive pairs (Eq. 21-22)
        """
        def compute_distribution_vector(embeddings, edges):
            vectors = []
            for u, v in edges:
                sim = torch.dot(embeddings[u], embeddings[v]) / tau
                # Compute softmax over all positive pairs for this user
                user_edges = [(u, v_j) for (u_i, v_j) in edges if u_i == u]
                if len(user_edges) > 1:
                    similarities = []
                    for u_i, v_j in user_edges:
                        sim_j = torch.dot(embeddings[u_i], embeddings[v_j]) / tau
                        similarities.append(sim_j)
                    similarities = torch.stack(similarities)
                    softmax_probs = F.softmax(similarities, dim=0)
                    # Find index of current edge
                    idx = user_edges.index((u, v))
                    vectors.append(torch.log(softmax_probs[idx] + 1e-8))
            return torch.stack(vectors) if vectors else torch.tensor(0.0)
        
        orig_dist = compute_distribution_vector(original_embeddings, remaining_edges)
        updated_dist = compute_distribution_vector(updated_embeddings, remaining_edges)
        
        if isinstance(orig_dist, torch.Tensor) and isinstance(updated_dist, torch.Tensor):
            return F.mse_loss(updated_dist, orig_dist)
        return torch.tensor(0.0)
    
    def contrast_loss(self, H, A_delta, dropout_rate=0.1):
        """
        Contrastive loss for influence generalization (Eq. 23-24)
        """
        # Apply dropout to A_delta
        mask = torch.rand_like(A_delta) > dropout_rate
        A_delta_prime = A_delta * mask.float()
        
        # Compute degree matrix for dropped version
        degree_prime = torch.sum(A_delta_prime, dim=1)
        D_delta_prime_sqrt_inv = torch.diag(1.0 / torch.sqrt(degree_prime + 1e-8))
        A_norm_prime = D_delta_prime_sqrt_inv @ A_delta_prime @ D_delta_prime_sqrt_inv
        
        # Compute H' with dropped adjacency
        H_prime = A_norm_prime @ H
        
        # Contrastive alignment (simplified)
        cos_sim = F.cosine_similarity(H, H_prime, dim=1)
        loss = -torch.log(torch.sigmoid(cos_sim / self.temperature)).mean()
        
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