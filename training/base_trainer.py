import torch
import torch.optim as optim
from tqdm import tqdm

class BaseModelTrainer:
    """Trainer for base recommendation model (before unlearning)"""
    
    def __init__(self, model, device='cuda', learning_rate=0.001, reg_weight=1e-4):
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.reg_weight = reg_weight
        
    def train(self, train_edges_tensor, num_users, num_items, 
              num_epochs=50, batch_size=2048, negative_samples=1):
        """
        Train the base GNN model using BPR loss
        
        Args:
            train_edges_tensor: Tensor of training edges [num_edges, 2]
            num_users: Number of users
            num_items: Number of items
            num_epochs: Training epochs
            batch_size: Batch size for training
            negative_samples: Number of negative samples per positive
        """
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Build sparse adjacency matrix once
        num_nodes = num_users + num_items
        if len(train_edges_tensor) > 0:
            users = train_edges_tensor[:, 0]
            items = train_edges_tensor[:, 1] + num_users
            
            # Create sparse COO format for memory efficiency
            edge_index = torch.stack([
                torch.cat([users, items]),
                torch.cat([items, users])
            ], dim=0)
            edge_values = torch.ones(edge_index.size(1), device=self.device)
            adj = torch.sparse_coo_tensor(
                edge_index, edge_values, (num_nodes, num_nodes), device=self.device
            ).coalesce()
        else:
            adj = torch.sparse_coo_tensor(
                torch.zeros(2, 0, dtype=torch.long, device=self.device),
                torch.zeros(0, device=self.device),
                (num_nodes, num_nodes)
            )
        
        # Compute embeddings once before training (for faster first batch)
        with torch.no_grad():
            _ = self.model(adj)
        
        history = {'loss': [], 'bpr_loss': [], 'reg_loss': []}
        
        print(f"Training base model for {num_epochs} epochs...")
        for epoch in tqdm(range(num_epochs), desc="Base Model Training"):
            self.model.train()
            epoch_loss = 0
            epoch_bpr = 0
            epoch_reg = 0
            
            # Shuffle training edges
            perm = torch.randperm(len(train_edges_tensor), device=self.device)
            train_edges_shuffled = train_edges_tensor[perm]
            
            num_batches = (len(train_edges_tensor) + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(train_edges_tensor))
                batch_edges = train_edges_shuffled[start_idx:end_idx]
                
                users = batch_edges[:, 0]
                pos_items = batch_edges[:, 1]
                
                # Sample negative items
                neg_items = torch.randint(0, num_items, (len(users), negative_samples), 
                                         device=self.device)
                
                # Forward pass
                embeddings = self.model(adj)
                user_emb = embeddings[users]
                pos_item_emb = embeddings[pos_items + num_users]
                neg_item_emb = embeddings[neg_items + num_users]
                
                # BPR loss
                pos_scores = (user_emb * pos_item_emb).sum(dim=1, keepdim=True)
                neg_scores = (user_emb.unsqueeze(1) * neg_item_emb).sum(dim=2)
                
                bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
                
                # L2 regularization
                reg_loss = self.reg_weight * (
                    user_emb.norm(2).pow(2) + 
                    pos_item_emb.norm(2).pow(2) + 
                    neg_item_emb.norm(2).pow(2)
                ) / len(users)
                
                loss = bpr_loss + reg_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_bpr += bpr_loss.item()
                epoch_reg += reg_loss.item()
            
            # Record history
            avg_loss = epoch_loss / num_batches
            avg_bpr = epoch_bpr / num_batches
            avg_reg = epoch_reg / num_batches
            
            history['loss'].append(avg_loss)
            history['bpr_loss'].append(avg_bpr)
            history['reg_loss'].append(avg_reg)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}: Loss={avg_loss:.4f}, "
                      f"BPR={avg_bpr:.4f}, Reg={avg_reg:.4f}")
        
        return history
