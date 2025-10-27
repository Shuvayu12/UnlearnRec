import yaml
import os
from typing import Dict, Any

class Config:
    """Configuration management class"""
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        if config_dict is None:
            config_dict = {}
        self._config = config_dict
    
    @classmethod
    def from_yaml(cls, file_path: str):
        """Load configuration from YAML file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Config file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(config_dict)
    
    def to_yaml(self, file_path: str):
        """Save configuration to YAML file"""
        with open(file_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        self._config[key] = value
    
    def update(self, new_config: Dict[str, Any]):
        """Update configuration with new values"""
        self._config.update(new_config)
    
    def __getitem__(self, key):
        return self._config[key]
    
    def __setitem__(self, key, value):
        self._config[key] = value
    
    def __contains__(self, key):
        return key in self._config
    
    def __repr__(self):
        return f"Config({self._config})"

def get_default_config():
    """Get default configuration"""
    return {
        'model': {
            'embedding_dim': 64,
            'num_layers': 3,
            'num_layers_ie': 3,
            'num_layers_mlp': 2
        },
        'training': {
            'num_pretrain_epochs': 100,
            'num_finetune_epochs': 3,
            'learning_rate': 0.001,
            'batch_size': 512,
            'unlearn_ratio': 0.05
        },
        'loss': {
            'lambda_u': 1.0,
            'lambda_p': 1.0,
            'lambda_c': 0.01,
            'temperature': 1.0
        },
        'evaluation': {
            'k_values': [10, 20, 50],
            'test_ratio': 0.2
        }
    }