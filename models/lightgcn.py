import torch
import torch.nn as nn
from .base_gnn import BaseGNN

class LightGCN(BaseGNN):
    """LightGCN implementation"""
    
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
        super(LightGCN, self).__init__(num_users, num_items, embedding_dim, num_layers)
    
    def forward(self, adjacency_matrix):
        return self.forward_with_embeddings(self.get_initial_embeddings(), adjacency_matrix)
    
    def forward_with_embeddings(self, initial_embeddings, adjacency_matrix):
        # LightGCN propagation (Eq. 1-3)
        embeddings = [initial_embeddings]
        
        for layer in range(self.num_layers):
            emb = adjacency_matrix @ embeddings[-1]
            embeddings.append(emb)
        
        # Combine all layers
        final_embeddings = torch.stack(embeddings, dim=0).mean(dim=0)
        return final_embeddings