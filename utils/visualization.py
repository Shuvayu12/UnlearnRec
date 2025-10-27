import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from typing import Dict, List

class Visualization:
    """Visualization utilities for UnlearnRec"""
    
    @staticmethod
    def plot_training_history(history: Dict[str, List[float]], save_path: str = None):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        metrics = ['total_loss', 'model_loss', 'unlearning_loss', 'preserving_loss']
        titles = ['Total Loss', 'Model Loss', 'Unlearning Loss', 'Preserving Loss']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            if metric in history:
                axes[i].plot(history[metric])
                axes[i].set_title(title)
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel('Loss')
                axes[i].grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_embedding_distribution(original_embeddings: torch.Tensor, 
                                  unlearned_embeddings: torch.Tensor,
                                  save_path: str = None):
        """Plot embedding distribution before and after unlearning"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Flatten embeddings for histogram
        orig_flat = original_embeddings.flatten().detach().cpu().numpy()
        unlearn_flat = unlearned_embeddings.flatten().detach().cpu().numpy()
        
        # Plot histograms
        axes[0].hist(orig_flat, bins=50, alpha=0.7, label='Original', color='blue')
        axes[0].set_title('Original Embedding Distribution')
        axes[0].set_xlabel('Embedding Value')
        axes[0].set_ylabel('Frequency')
        
        axes[1].hist(unlearn_flat, bins=50, alpha=0.7, label='Unlearned', color='red')
        axes[1].set_title('Unlearned Embedding Distribution')
        axes[1].set_xlabel('Embedding Value')
        axes[1].set_ylabel('Frequency')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_score_distribution(original_scores: List[float], 
                              unlearned_scores: List[float],
                              adversarial_scores: List[float],
                              save_path: str = None):
        """Plot score distribution for different edge types"""
        plt.figure(figsize=(10, 6))
        
        # Create box plot
        data = [original_scores, unlearned_scores, adversarial_scores]
        labels = ['Original', 'Unlearned', 'Adversarial']
        
        box_plot = plt.boxplot(data, labels=labels, patch_artist=True)
        
        # Customize colors
        colors = ['lightblue', 'lightcoral', 'lightgreen']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        plt.title('Prediction Score Distribution')
        plt.ylabel('Score')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_efficiency_comparison(methods: List[str], 
                                 times: List[float],
                                 memory_usage: List[float],
                                 save_path: str = None):
        """Plot efficiency comparison between methods"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Time comparison
        bars1 = ax1.bar(methods, times, color='skyblue', alpha=0.7)
        ax1.set_title('Processing Time Comparison')
        ax1.set_ylabel('Time (seconds)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}s', ha='center', va='bottom')
        
        # Memory comparison
        bars2 = ax2.bar(methods, memory_usage, color='lightcoral', alpha=0.7)
        ax2.set_title('Memory Usage Comparison')
        ax2.set_ylabel('Memory (GB)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}GB', ha='center', va='bottom')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_unlearning_ratio_study(ratios: List[float], 
                                  recalls: List[float],
                                  ndcgs: List[float],
                                  mi_ngs: List[float],
                                  save_path: str = None):
        """Plot unlearning ratio study results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Utility metrics
        ax1.plot(ratios, recalls, 'o-', label='Recall', linewidth=2, markersize=8)
        ax1.plot(ratios, ndcgs, 's-', label='NDCG', linewidth=2, markersize=8)
        ax1.set_xlabel('Unlearning Ratio')
        ax1.set_ylabel('Score')
        ax1.set_title('Utility vs Unlearning Ratio')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Unlearning efficacy
        ax2.plot(ratios, mi_ngs, '^-', color='red', label='MI-NG', linewidth=2, markersize=8)
        ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Threshold')
        ax2.set_xlabel('Unlearning Ratio')
        ax2.set_ylabel('MI-NG Score')
        ax2.set_title('Unlearning Efficacy vs Ratio')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()