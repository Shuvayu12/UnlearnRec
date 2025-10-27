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
        nn.init.normal_(self.W_eta, mean=0.0, std=0.01)
        
        # Initialize MLP with identity-like transformation
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.eye_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, A_delta, E_original):
        """
        Args:
            A_delta: Influence Dependency Matrix [num_nodes, num_nodes]
            E_original: Original embeddings [num_nodes, embedding_dim]
        Returns:
            E0_updated: Updated 0-layer embeddings [num_nodes, embedding_dim]
        """
        # Compute degree matrix for A_delta
        degree = torch.sum(A_delta, dim=1)
        D_delta_sqrt_inv = torch.diag(1.0 / torch.sqrt(degree + 1e-8))
        
        # Normalized adjacency (Eq. 15)
        A_norm = D_delta_sqrt_inv @ A_delta @ D_delta_sqrt_inv
        
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
        
        return E0_updated
    
    def freeze_pretrained_params(self):
        """Freeze H0 and W_eta after pre-training"""
        self.H0.requires_grad = False
        self.W_eta.requires_grad = False
    
    def unfreeze_mlp(self):
        """Unfreeze MLP for fine-tuning"""
        for param in self.mlp.parameters():
            param.requires_grad = True