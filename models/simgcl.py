import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_gnn import BaseGNN

class SimGCL(BaseGNN):
    """SimGCL implementation with simple contrastive learning"""
    
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3, 
                 contrastive_temp=0.2, noise_eps=0.1):
        super(SimGCL, self).__init__(num_users, num_items, embedding_dim, num_layers)
        self.contrastive_temp = contrastive_temp
        self.noise_eps = noise_eps
        
    def forward(self, adjacency_matrix):
        return self.forward_with_embeddings(self.get_initial_embeddings(), adjacency_matrix)
    
    def forward_with_embeddings(self, initial_embeddings, adjacency_matrix):
        # LightGCN propagation
        embeddings = [initial_embeddings]
        
        for layer in range(self.num_layers):
            emb = adjacency_matrix @ embeddings[-1]
            embeddings.append(emb)
        
        # Combine all layers
        final_embeddings = torch.stack(embeddings, dim=0).mean(dim=0)
        return final_embeddings
    
    def compute_ssl_loss(self, embeddings):
        """Compute SimGCL contrastive loss with noise-based augmentation"""
        # Create two views by adding different noise
        view1 = self._add_controlled_noise(embeddings)
        view2 = self._add_controlled_noise(embeddings)
        
        # Normalize embeddings
        view1 = F.normalize(view1, p=2, dim=1)
        view2 = F.normalize(view2, p=2, dim=1)
        
        # Compute contrastive loss
        loss = self._info_nce_loss(view1, view2)
        return loss
    
    def _add_controlled_noise(self, embeddings):
        """Add controlled noise to embeddings"""
        noise = torch.randn_like(embeddings) * self.noise_eps
        return embeddings + noise
    
    def _info_nce_loss(self, view1, view2):
        """InfoNCE loss for contrastive learning"""
        batch_size = view1.size(0)
        
        # Compute similarity matrix
        similarity = torch.matmul(view1, view2.T) / self.contrastive_temp
        
        # Positive pairs are diagonal elements
        labels = torch.arange(batch_size, device=view1.device)
        
        loss = F.cross_entropy(similarity, labels)
        return loss