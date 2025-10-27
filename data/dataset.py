import torch
from torch.utils.data import Dataset
import numpy as np

class RecommendationDataset(Dataset):
    """Dataset for recommendation systems"""
    
    def __init__(self, user_item_interactions, num_users, num_items):
        self.user_item_interactions = user_item_interactions  # List of (user, item) tuples
        self.num_users = num_users
        self.num_items = num_items
        self.all_edges = user_item_interactions
    
    def __len__(self):
        return len(self.user_item_interactions)
    
    def __getitem__(self, idx):
        user, item = self.user_item_interactions[idx]
        return torch.tensor(user), torch.tensor(item)
    
    def get_adjacency_matrix(self, edges=None):
        """Get adjacency matrix for given edges"""
        if edges is None:
            edges = self.all_edges
            
        num_nodes = self.num_users + self.num_items
        adj = torch.zeros((num_nodes, num_nodes))
        
        for u, i in edges:
            i_node = i + self.num_users
            adj[u, i_node] = 1
            adj[i_node, u] = 1
            
        return adj
    
    def split_edges(self, test_ratio=0.2):
        """Split edges into train and test sets"""
        num_test = int(len(self.all_edges) * test_ratio)
        indices = torch.randperm(len(self.all_edges))
        
        test_indices = indices[:num_test]
        train_indices = indices[num_test:]
        
        train_edges = [self.all_edges[i] for i in train_indices]
        test_edges = [self.all_edges[i] for i in test_indices]
        
        return train_edges, test_edges