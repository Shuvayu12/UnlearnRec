import torch
import torch.nn as nn
import torch.nn.functional as F

class InfluenceEncoder(nn.Module):
    """Influence Encoder for recommendation unlearning"""
    
    def __init__(self, num_nodes, embedding_dim, num_layers_ie=3, num_layers_mlp=2):
        super(InfluenceEncoder, self).__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.num_layers_ie = num_layers_ie
        self.num_layers_mlp = num_layers_mlp
        
        # Trainable parameters
        self.H0 = nn.Parameter(torch.zeros(num_nodes, embedding_dim))
        self.W_eta = nn.Parameter(torch.zeros(num_nodes, 1))
        
        # MLP for final transformation
        self.mlp = self._build_mlp()
        
        self.reset_parameters()
    
    def _build_mlp(self):
        layers = []
        input_dim = self.embedding_dim
        for i in range(self.num_layers_mlp - 1):
            layers.append(nn.Linear(input_dim, input_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(input_dim, self.embedding_dim))
        return nn.Sequential(*layers)
    
    def reset_parameters(self):
        # Initialize with small values around 0 as mentioned in paper
        nn.init.normal_(self.H0, mean=0.0, std=0.01)
        nn.init.constant_(self.W_eta, 0.01)  # Small constant instead of normal
        
        # Initialize MLP layers properly
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                # Use Xavier initialization for better stability
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, A_delta, E_original):
        """
        Args:
            A_delta: Influence Dependency Matrix [num_nodes, num_nodes]
            E_original: Original embeddings [num_nodes, embedding_dim]
        Returns:
            E0_updated: Updated 0-layer embeddings [num_nodes, embedding_dim]
        """
        # Check if A_delta is empty or all zeros
        if A_delta.is_sparse:
            degree = torch.sparse.sum(A_delta, dim=1).to_dense()
        else:
            degree = torch.sum(A_delta, dim=1)
        
        # If no edges, return original embeddings
        if degree.sum() == 0:
            return E_original.clone()
        
        # Safe degree normalization with better epsilon handling
        degree_inv_sqrt = torch.pow(degree + 1e-10, -0.5)
        degree_inv_sqrt = torch.clamp(degree_inv_sqrt, min=0.0, max=1e10)
        degree_inv_sqrt[torch.isnan(degree_inv_sqrt)] = 0.0
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        
        # Normalized adjacency using element-wise multiplication
        A_norm = A_delta * degree_inv_sqrt.view(-1, 1) * degree_inv_sqrt.view(1, -1)
        
        # Check for NaN in normalized matrix
        if torch.isnan(A_norm).any():
            print("Warning: NaN detected in normalized adjacency matrix")
            A_norm = torch.nan_to_num(A_norm, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Compute H (Influence Estimation Matrix)
        H = self.H0
        for _ in range(self.num_layers_ie):
            H = A_norm @ H
        
        H_final = H  # Readout IEM
        
        # Compute E_w (Eq. 16)
        E_w = E_original * self.W_eta
        for _ in range(self.num_layers_ie):
            E_w = A_norm @ E_w
        
        # Combine and apply MLP (Eq. 17)
        delta_E0 = -E_w + H_final
        delta_E0_processed = self.mlp(delta_E0)
        E0_updated = delta_E0_processed + E_original
        
        # Safety check for NaN
        if torch.isnan(E0_updated).any():
            print("Warning: NaN detected in updated embeddings, returning original")
            return E_original.clone()
        
        return E0_updated
    
    def freeze_pretrained_params(self):
        """Freeze H0 and W_eta after pre-training"""
        self.H0.requires_grad = False
        self.W_eta.requires_grad = False
    
    def unfreeze_mlp(self):
        """Unfreeze MLP for fine-tuning"""
        for param in self.mlp.parameters():
            param.requires_grad = True