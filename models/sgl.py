import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_gnn import BaseGNN

class SGL(BaseGNN):
    """SGL (Self-supervised Graph Learning) implementation"""
    
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3, 
                 contrastive_temp=0.2, drop_ratio=0.1):
        super(SGL, self).__init__(num_users, num_items, embedding_dim, num_layers)
        self.contrastive_temp = contrastive_temp
        self.drop_ratio = drop_ratio
        
    def forward(self, adjacency_matrix):
        return self.forward_with_embeddings(self.get_initial_embeddings(), adjacency_matrix)
    
    def forward_with_embeddings(self, initial_embeddings, adjacency_matrix):
        # LightGCN-like propagation
        embeddings = [initial_embeddings]
        
        for layer in range(self.num_layers):
            emb = adjacency_matrix @ embeddings[-1]
            embeddings.append(emb)
        
        # Combine all layers
        final_embeddings = torch.stack(embeddings, dim=0).mean(dim=0)
        return final_embeddings
    
    def compute_ssl_loss(self, embeddings):
        """Compute self-supervised contrastive loss"""
        # Create two views through graph augmentation
        view1 = self._create_augmented_view(embeddings)
        view2 = self._create_augmented_view(embeddings)
        
        # Compute contrastive loss between views
        loss = self._contrastive_loss(view1, view2)
        return loss
    
    def _create_augmented_view(self, embeddings):
        """Create augmented view through node dropout"""
        batch_size = embeddings.size(0)
        drop_mask = torch.rand(batch_size, device=embeddings.device) > self.drop_ratio
        augmented_emb = embeddings * drop_mask.unsqueeze(1).float()
        return F.normalize(augmented_emb, p=2, dim=1)
    
    def _contrastive_loss(self, view1, view2):
        """Compute contrastive loss between two views"""
        batch_size = view1.size(0)
        
        # Compute similarity matrix
        similarity = torch.matmul(view1, view2.T) / self.contrastive_temp
        
        # Positive pairs are diagonal elements
        pos_sim = torch.diag(similarity)
        
        # Negative pairs are off-diagonal elements
        neg_sim = similarity[~torch.eye(batch_size, dtype=bool)].view(batch_size, batch_size-1)
        
        # Compute loss
        numerator = torch.exp(pos_sim)
        denominator = numerator + torch.sum(torch.exp(neg_sim), dim=1)
        loss = -torch.log(numerator / denominator).mean()
        
        return loss