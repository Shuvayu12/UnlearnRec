import torch
import torch.optim as optim
from tqdm import tqdm

class FineTuner:
    """Fine-tuning module for UnlearnRec"""
    
    def __init__(self, model, influence_encoder, loss_fn, device='cuda'):
        self.model = model
        self.influence_encoder = influence_encoder
        self.loss_fn = loss_fn
        self.device = device
        
    def fine_tune(self, unlearn_edges, all_edges, num_users, num_items, 
                  num_epochs=3, learning_rate=0.001):
        """
        Fine-tune the influence encoder and model for specific unlearning request
        """
        # Construct matrices
        A_delta = self._construct_A_delta(unlearn_edges, num_users, num_items)
        remaining_edges = [e for e in all_edges if e not in unlearn_edges]
        A_r = self._construct_A_r(remaining_edges, num_users, num_items)
        
        # Get original embeddings
        E_original = self.model.get_initial_embeddings()
        
        # Initial forward pass
        with torch.no_grad():
            E0_initial = self.influence_encoder(A_delta, E_original)
        
        # Make embeddings trainable for fine-tuning
        E0_updated = E0_initial.clone().requires_grad_(True)
        
        # Unfreeze MLP for fine-tuning
        self.influence_encoder.unfreeze_mlp()
        
        # Optimizer for embeddings and MLP
        optimizer = optim.Adam([
            {'params': E0_updated, 'lr': learning_rate},
            {'params': self.influence_encoder.mlp.parameters(), 'lr': learning_rate}
        ])
        
        history = []
        
        for epoch in tqdm(range(num_epochs), desc="Fine-tuning"):
            optimizer.zero_grad()
            
            # Compute predictions with current embeddings
            predictions = self.model.predict(E0_updated, A_r)
            
            # Compute fine-tuning loss (L_M + lambda_u * L_u as mentioned in paper)
            model_loss, embeddings_final = self.loss_fn.model_loss(
                self.model, E0_updated, A_r, unlearn_edges
            )
            
            unlearning_loss = self.loss_fn.unlearning_loss(predictions, unlearn_edges)
            
            total_loss = model_loss + self.loss_fn.lambda_u * unlearning_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            history.append({
                'epoch': epoch,
                'total_loss': total_loss.item(),
                'model_loss': model_loss.item(),
                'unlearning_loss': unlearning_loss.item()
            })
            
            if epoch % 1 == 0:
                print(f"Fine-tune Epoch {epoch}: Total Loss: {total_loss.item():.4f}")
        
        return E0_updated.detach(), history
    
    def _construct_A_delta(self, unlearn_edges, num_users, num_items):
        """Construct influence dependency matrix"""
        num_nodes = num_users + num_items
        A_delta = torch.zeros((num_nodes, num_nodes), device=self.device)
        
        for u, v in unlearn_edges:
            v_node = v + num_users
            A_delta[u, v_node] = 1
            A_delta[v_node, u] = 1
            
        return A_delta
    
    def _construct_A_r(self, remaining_edges, num_users, num_items):
        """Construct residual adjacency matrix"""
        num_nodes = num_users + num_items
        A_r = torch.zeros((num_nodes, num_nodes), device=self.device)
        
        for u, v in remaining_edges:
            v_node = v + num_users
            A_r[u, v_node] = 1
            A_r[v_node, u] = 1
            
        return A_r