import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseGNN(nn.Module):
    """Base class for GNN-based recommender models"""
    
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
        super(BaseGNN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
    
    def get_initial_embeddings(self):
        """Get concatenated initial embeddings for all nodes"""
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        return torch.cat([user_emb, item_emb], dim=0)
    
    def forward(self, adjacency_matrix):
        """Standard forward pass"""
        # This should be implemented by specific GNN models
        raise NotImplementedError
    
    def forward_with_embeddings(self, initial_embeddings, adjacency_matrix):
        """Forward pass with custom initial embeddings"""
        # This should be implemented by specific GNN models
        raise NotImplementedError
    
    def predict(self, initial_embeddings, adjacency_matrix):
        """Make predictions with custom embeddings"""
        final_embeddings = self.forward_with_embeddings(initial_embeddings, adjacency_matrix)
        user_emb, item_emb = self._split_embeddings(final_embeddings)
        return user_emb @ item_emb.t()
    
    def compute_bpr_loss(self, predictions, positive_edges):
        """Compute BPR loss"""
        loss = 0
        for u, i in positive_edges:
            # Sample negative item
            j = torch.randint(0, self.num_items, (1,)).item()
            x_ui = predictions[u, i]
            x_uj = predictions[u, j]
            loss += -torch.log(torch.sigmoid(x_ui - x_uj))
        
        return loss / len(positive_edges) if len(positive_edges) > 0 else torch.tensor(0.0)
    
    def _split_embeddings(self, embeddings):
        """Split concatenated embeddings into user and item embeddings"""
        user_emb = embeddings[:self.num_users]
        item_emb = embeddings[self.num_users:]
        return user_emb, item_emb
    
    def clone(self):
        """Create a copy of the model"""
        return type(self)(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=self.embedding_dim,
            num_layers=self.num_layers
        )
    
    def update_embeddings(self, new_embeddings, adjacency_matrix):
        """Update model with new embeddings"""
        user_emb, item_emb = self._split_embeddings(new_embeddings)
        self.user_embedding.weight.data = user_emb
        self.item_embedding.weight.data = item_emb