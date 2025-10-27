import torch
import numpy as np
from typing import List, Tuple, Dict

class UnlearningManager:
    """Manager for the complete unlearning pipeline"""
    
    def __init__(self, model, influence_encoder, device='cuda'):
        self.model = model
        self.influence_encoder = influence_encoder
        self.device = device
        
    def construct_influence_dependency_matrix(self, unlearn_edges: List[Tuple[int, int]], 
                                           num_users: int, num_items: int) -> torch.Tensor:
        """
        Construct Influence Dependency Matrix from unlearning edges
        """
        num_nodes = num_users + num_items
        A_delta = torch.zeros((num_nodes, num_nodes), device=self.device)
        
        for u, v in unlearn_edges:
            # Convert item index to node index
            v_node = v + num_users
            A_delta[u, v_node] = 1
            A_delta[v_node, u] = 1  # Symmetric
            
        return A_delta
    
    def construct_residual_adjacency(self, all_edges: List[Tuple[int, int]], 
                                   unlearn_edges: List[Tuple[int, int]],
                                   num_users: int, num_items: int) -> torch.Tensor:
        """
        Construct residual adjacency matrix from remaining edges
        """
        num_nodes = num_users + num_items
        A_r = torch.zeros((num_nodes, num_nodes), device=self.device)
        
        remaining_edges = [edge for edge in all_edges if edge not in unlearn_edges]
        
        for u, v in remaining_edges:
            v_node = v + num_users
            A_r[u, v_node] = 1
            A_r[v_node, u] = 1  # Symmetric
            
        return A_r
    
    def process_unlearning_request(self, unlearn_edges: List[Tuple[int, int]], 
                                 all_edges: List[Tuple[int, int]],
                                 num_users: int, num_items: int,
                                 fine_tune: bool = False, 
                                 fine_tune_epochs: int = 3) -> Dict:
        """
        Process unlearning request using the pre-trained influence encoder
        """
        # Construct matrices
        A_delta = self.construct_influence_dependency_matrix(unlearn_edges, num_users, num_items)
        A_r = self.construct_residual_adjacency(all_edges, unlearn_edges, num_users, num_items)
        
        # Get original embeddings
        E_original = self.model.get_initial_embeddings()
        
        # Apply influence encoder
        with torch.no_grad():
            E0_updated = self.influence_encoder(A_delta, E_original)
        
        # Fine-tuning if requested
        if fine_tune:
            E0_updated = self._fine_tune(unlearn_edges, all_edges, A_delta, A_r, 
                                       E_original, E0_updated, fine_tune_epochs)
        
        # Update model with new embeddings
        unlearned_model = self._create_unlearned_model(E0_updated, A_r)
        
        return {
            'unlearned_model': unlearned_model,
            'updated_embeddings': E0_updated,
            'A_delta': A_delta,
            'A_r': A_r
        }
    
    def _fine_tune(self, unlearn_edges, all_edges, A_delta, A_r, 
                  E_original, E0_initial, epochs):
        """Fine-tuning process"""
        # Implementation of fine-tuning as described in paper
        # This would involve optimizing L_M + lambda_u * L_u
        # Simplified implementation
        E0_updated = E0_initial.clone().requires_grad_(True)
        
        optimizer = torch.optim.Adam([E0_updated], lr=0.001)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Compute predictions with current embeddings
            predictions = self.model.predict(E0_updated, A_r)
            
            # Compute unlearning loss
            loss = 0
            for u, v in unlearn_edges:
                score = predictions[u, v]
                loss += -torch.log(torch.sigmoid(-score))
            
            loss = loss / len(unlearn_edges)
            loss.backward()
            optimizer.step()
        
        return E0_updated.detach()
    
    def _create_unlearned_model(self, E0_updated, A_r):
        """Create unlearned model with updated embeddings"""
        # Create a copy of the original model with updated embeddings
        unlearned_model = self.model.clone()
        unlearned_model.update_embeddings(E0_updated, A_r)
        return unlearned_model