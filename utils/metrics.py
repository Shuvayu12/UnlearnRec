import torch
import numpy as np
from sklearn.metrics import roc_auc_score

class UnlearningMetrics:
    """Metrics for evaluating unlearning performance"""
    
    @staticmethod
    def membership_inference_bf(model, unlearn_edges, original_scores):
        """
        MI-BF: Ratio of average probability before and after unlearning
        """
        with torch.no_grad():
            # Get scores after unlearning
            current_scores = []
            for u, v in unlearn_edges:
                score = model.predict_single(u, v)
                current_scores.append(score.item())
            
            avg_before = np.mean([score for u, v, score in original_scores])
            avg_after = np.mean(current_scores)
            
            return avg_before / avg_after if avg_after > 0 else float('inf')
    
    @staticmethod
    def membership_inference_ng(model, unlearn_edges, negative_edges):
        """
        MI-NG: Ratio between unlearned edges and negative samples
        """
        with torch.no_grad():
            # Scores for unlearned edges
            unlearn_scores = []
            for u, v in unlearn_edges:
                score = model.predict_single(u, v)
                unlearn_scores.append(score.item())
            
            # Scores for negative edges
            neg_scores = []
            for u, v in negative_edges:
                score = model.predict_single(u, v)
                neg_scores.append(score.item())
            
            avg_unlearn = np.mean(unlearn_scores)
            avg_neg = np.mean(neg_scores)
            
            return avg_neg / avg_unlearn if avg_unlearn > 0 else float('inf')
    
    @staticmethod
    def recall_at_k(model, test_edges, k=20):
        """Recall@K metric"""
        recalls = []
        for user in set(u for u, _ in test_edges):
            user_items = [v for u, v in test_edges if u == user]
            if not user_items:
                continue
                
            # Get top-K recommendations for user
            scores = model.predict_user(user)
            top_k_items = torch.topk(scores, k=k).indices.tolist()
            
            # Compute recall
            hit = len(set(user_items) & set(top_k_items))
            recall = hit / len(user_items)
            recalls.append(recall)
        
        return np.mean(recalls) if recalls else 0.0
    
    @staticmethod
    def ndcg_at_k(model, test_edges, k=20):
        """NDCG@K metric"""
        ndcgs = []
        for user in set(u for u, _ in test_edges):
            user_items = [v for u, v in test_edges if u == user]
            if not user_items:
                continue
                
            # Get top-K recommendations for user
            scores = model.predict_user(user)
            top_k_items = torch.topk(scores, k=k).indices.tolist()
            
            # Compute DCG and IDCG
            dcg = 0
            idcg = 0
            
            for i, item in enumerate(top_k_items):
                if item in user_items:
                    dcg += 1 / np.log2(i + 2)
            
            for i in range(min(len(user_items), k)):
                idcg += 1 / np.log2(i + 2)
            
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcgs.append(ndcg)
        
        return np.mean(ndcgs) if ndcgs else 0.0

class AttackMetrics:
    """Metrics for adversarial attack evaluation"""
    
    @staticmethod
    def adversarial_edge_detection(model, adversarial_edges, threshold=0.5):
        """
        Detect if adversarial edges are properly unlearned
        """
        detected = 0
        for u, v in adversarial_edges:
            score = model.predict_single(u, v)
            if score < threshold:
                detected += 1
        
        return detected / len(adversarial_edges) if adversarial_edges else 0.0
    
    @staticmethod
    def embedding_similarity(original_embeddings, unlearned_embeddings):
        """
        Compute cosine similarity between original and unlearned embeddings
        """
        orig_norm = torch.nn.functional.normalize(original_embeddings, p=2, dim=1)
        unlearn_norm = torch.nn.functional.normalize(unlearned_embeddings, p=2, dim=1)
        
        similarity = torch.sum(orig_norm * unlearn_norm, dim=1)
        return similarity.mean().item()