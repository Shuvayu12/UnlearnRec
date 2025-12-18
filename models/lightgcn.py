import torch
import torch.nn as nn
from .base_gnn import BaseGNN

class LightGCN(BaseGNN):
    """LightGCN implementation"""
    
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
        super(LightGCN, self).__init__(num_users, num_items, embedding_dim, num_layers)
        self._cached_norm_adj = None
        self._cached_adj_hash = None
    
    def _normalize_adjacency(self, adjacency_matrix):
        """Normalize adjacency matrix by degree (D^-0.5 * A * D^-0.5)"""
        # Create a hash for caching (handle both sparse and dense tensors)
        try:
            if adjacency_matrix.is_sparse:
                # For sparse tensors, use shape and number of nonzeros
                adj_hash = hash((adjacency_matrix.shape, adjacency_matrix._nnz(), id(adjacency_matrix)))
            else:
                adj_hash = hash(adjacency_matrix.data_ptr())
        except:
            # Fallback: disable caching for this matrix
            adj_hash = None
        
        if adj_hash is not None and self._cached_norm_adj is not None and self._cached_adj_hash == adj_hash:
            return self._cached_norm_adj
        
        # Use sparse format if matrix is sparse enough
        if adjacency_matrix.is_sparse:
            # Sparse normalization
            row_sum = torch.sparse.sum(adjacency_matrix, dim=1).to_dense()
            degree_inv_sqrt = torch.pow(row_sum + 1e-10, -0.5)
            degree_inv_sqrt = torch.clamp(degree_inv_sqrt, min=0.0, max=1e10)
            degree_inv_sqrt[torch.isnan(degree_inv_sqrt)] = 0.0
            degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
            
            # Normalize using element-wise multiplication (avoid dense diagonal)
            indices = adjacency_matrix.indices()
            values = adjacency_matrix.values()
            
            # Apply D^-0.5 from both sides
            normalized_values = values * degree_inv_sqrt[indices[0]] * degree_inv_sqrt[indices[1]]
            normalized_adj = torch.sparse_coo_tensor(
                indices, normalized_values, adjacency_matrix.size(), device=adjacency_matrix.device
            ).coalesce()
        else:
            # Dense normalization (fallback)
            degree = torch.sum(adjacency_matrix, dim=1)
            degree_inv_sqrt = torch.pow(degree + 1e-10, -0.5)
            degree_inv_sqrt = torch.clamp(degree_inv_sqrt, min=0.0, max=1e10)
            degree_inv_sqrt[torch.isnan(degree_inv_sqrt)] = 0.0
            degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
            
            # Element-wise multiplication instead of diagonal matrix
            normalized_adj = adjacency_matrix * degree_inv_sqrt.view(-1, 1) * degree_inv_sqrt.view(1, -1)
        
        # Cache for reuse (only if hashing succeeded)
        if adj_hash is not None:
            self._cached_norm_adj = normalized_adj
            self._cached_adj_hash = adj_hash
        
        return normalized_adj
    
    def forward(self, adjacency_matrix):
        return self.forward_with_embeddings(self.get_initial_embeddings(), adjacency_matrix)
    
    def forward_with_embeddings(self, initial_embeddings, adjacency_matrix):
        # Normalize adjacency matrix (cached)
        normalized_adj = self._normalize_adjacency(adjacency_matrix)
        
        # LightGCN propagation (Eq. 1-3)
        embeddings = [initial_embeddings]
        
        for layer in range(self.num_layers):
            if normalized_adj.is_sparse:
                emb = torch.sparse.mm(normalized_adj, embeddings[-1])
            else:
                emb = normalized_adj @ embeddings[-1]
            embeddings.append(emb)
        
        # Combine all layers
        final_embeddings = torch.stack(embeddings, dim=0).mean(dim=0)
        return final_embeddings