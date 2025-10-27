import os
import zipfile
import gzip
import shutil
import urllib.request
import pandas as pd
import numpy as np
import torch
from typing import Tuple, List, Optional

# URLs for datasets
ML_1M_URL = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'
GOWALLA_URL = 'https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz'
YELP2018_URL = 'https://s3.amazonaws.com/yangdingqi-dataset/yelp_2018/yelp2018.csv'

class DataPreprocessor:
    """Data preprocessing utilities for recommendation datasets"""
    
    @staticmethod
    def load_movielens_1m(file_path: str) -> Tuple[List[Tuple[int, int]], int, int]:
        """Load MovieLens-1M dataset"""
        # Accept either path to ratings.dat or preprocessed csv
        if file_path.endswith('.zip'):
            # Extract ratings.dat into a temp dir
            with zipfile.ZipFile(file_path, 'r') as z:
                members = z.namelist()
                # ratings.dat is expected in zip
                target = None
                for m in members:
                    if m.endswith('ratings.dat') or m.endswith('ratings.csv'):
                        target = m
                        break
                if target is None:
                    # fallback to files containing 'ratings'
                    for m in members:
                        if 'rating' in m.lower():
                            target = m
                            break
                if target is None:
                    raise FileNotFoundError('ratings.dat not found in the zip archive')

                extract_dir = os.path.join(os.path.dirname(file_path), 'ml-1m-extracted')
                os.makedirs(extract_dir, exist_ok=True)
                z.extract(target, path=extract_dir)
                extracted_path = os.path.join(extract_dir, target)
                ratings = pd.read_csv(extracted_path, sep='::', engine='python', 
                                      names=['user_id', 'item_id', 'rating', 'timestamp'])
        else:
            ratings = pd.read_csv(file_path, sep='::', engine='python', 
                                  names=['user_id', 'item_id', 'rating', 'timestamp'])

        # Filter positive interactions (assuming rating >= 4 as positive)
        positive_interactions = ratings[ratings['rating'] >= 4].copy()

        # Map raw ids to contiguous 0-indexed ids using categorical encoding
        users = positive_interactions['user_id'].astype('category')
        items = positive_interactions['item_id'].astype('category')
        positive_interactions['u_idx'] = users.cat.codes
        positive_interactions['i_idx'] = items.cat.codes

        interactions = list(
            zip(
                positive_interactions['u_idx'].tolist(),
                positive_interactions['i_idx'].tolist()
            )
        )
        num_users = len(users.cat.categories)
        num_items = len(items.cat.categories)

        return interactions, num_users, num_items
    
    @staticmethod
    def load_gowalla(file_path: str) -> Tuple[List[Tuple[int, int]], int, int]:
        """Load Gowalla dataset"""
        # Gowalla checkins are typically in a tab-separated file with user_id, venue_id, lat, lon, time
        # We'll parse user and item (venue) IDs and map them to contiguous indices
        if file_path.endswith('.gz'):
            open_fn = gzip.open
        else:
            open_fn = open

        user_map = {}
        item_map = {}
        interactions = []

        with open_fn(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
            for line in f:
                cols = line.strip().split('\t')
                if len(cols) < 2:
                    continue
                try:
                    user_raw = cols[0]
                    item_raw = cols[1]
                except Exception:
                    continue

                if user_raw not in user_map:
                    user_map[user_raw] = len(user_map)
                if item_raw not in item_map:
                    item_map[item_raw] = len(item_map)

                interactions.append((user_map[user_raw], item_map[item_raw]))

        num_users = len(user_map)
        num_items = len(item_map)

        return interactions, num_users, num_items
    
    @staticmethod
    def load_yelp2018(file_path: str) -> Tuple[List[Tuple[int, int]], int, int]:
        """Load Yelp2018 dataset"""
        # Yelp2018 is often provided as a large CSV with user_id,item_id,rating,...
        df = pd.read_csv(file_path)

        if 'user_id' in df.columns and 'business_id' in df.columns:
            user_col = 'user_id'
            item_col = 'business_id'
        elif 'user_id' in df.columns and 'item_id' in df.columns:
            user_col = 'user_id'
            item_col = 'item_id'
        else:
            # Try first two columns
            user_col = df.columns[0]
            item_col = df.columns[1]

        # Map to contiguous indices
        users = df[user_col].astype('category')
        items = df[item_col].astype('category')

        df['u_idx'] = users.cat.codes
        df['i_idx'] = items.cat.codes

        # Consider positive interactions where rating >= 4 if rating exists
        if 'rating' in df.columns:
            pos = df[df['rating'] >= 4]
        else:
            pos = df

        interactions = list(zip(pos['u_idx'].tolist(), pos['i_idx'].tolist()))
        num_users = int(df['u_idx'].max()) + 1
        num_items = int(df['i_idx'].max()) + 1

        return interactions, num_users, num_items

    @staticmethod
    def download_url(url: str, dest: str, overwrite: bool = False) -> str:
        """Download a URL to destination path. Returns the path to the file."""
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.exists(dest) and not overwrite:
            return dest
        print(f"Downloading {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
        return dest

    @staticmethod
    def download_and_preprocess(dataset: str, dest_dir: str = 'data') -> Tuple[List[Tuple[int, int]], int, int]:
        """Download and preprocess a supported dataset into interactions.

        Supported: 'movielens-1m', 'gowalla', 'yelp2018'
        Returns: interactions, num_users, num_items
        """
        dataset = dataset.lower()
        os.makedirs(dest_dir, exist_ok=True)

        if dataset == 'movielens-1m':
            zip_path = os.path.join(dest_dir, 'ml-1m.zip')
            DataPreprocessor.download_url(ML_1M_URL, zip_path)
            # Use load_movielens_1m which can accept the zip path
            interactions, n_users, n_items = DataPreprocessor.load_movielens_1m(zip_path)
            # Save processed CSV for future use
            proc_path = os.path.join(dest_dir, 'movielens_1m_interactions.csv')
            pd.DataFrame(interactions, columns=['user_id', 'item_id']).to_csv(proc_path, index=False)
            return interactions, n_users, n_items

        elif dataset == 'gowalla':
            gz_path = os.path.join(dest_dir, 'gowalla_checkins.txt.gz')
            try:
                DataPreprocessor.download_url(GOWALLA_URL, gz_path)
            except Exception as e:
                raise RuntimeError(f'Failed to download Gowalla dataset: {e}\nPlease provide the file manually at {gz_path}')

            interactions, n_users, n_items = DataPreprocessor.load_gowalla(gz_path)
            proc_path = os.path.join(dest_dir, 'gowalla_interactions.csv')
            pd.DataFrame(interactions, columns=['user_id', 'item_id']).to_csv(proc_path, index=False)
            return interactions, n_users, n_items

        elif dataset == 'yelp2018':
            csv_path = os.path.join(dest_dir, 'yelp2018.csv')
            try:
                DataPreprocessor.download_url(YELP2018_URL, csv_path)
            except Exception as e:
                raise RuntimeError(f'Failed to download Yelp2018 dataset: {e}\nPlease provide the file manually at {csv_path}')

            interactions, n_users, n_items = DataPreprocessor.load_yelp2018(csv_path)
            proc_path = os.path.join(dest_dir, 'yelp2018_interactions.csv')
            pd.DataFrame(interactions, columns=['user_id', 'item_id']).to_csv(proc_path, index=False)
            return interactions, n_users, n_items

        else:
            raise ValueError(f'Unsupported dataset: {dataset}')

    @staticmethod
    def build_recommendation_dataset(dataset: str, dest_dir: str = 'data'):
        """Download/preprocess and build a RecommendationDataset instance.

        Returns a RecommendationDataset ready to be used by training code.
        """
        from data.dataset import RecommendationDataset

        interactions_csv = os.path.join(dest_dir, f'{dataset}_interactions.csv')
        if os.path.exists(interactions_csv):
            try:
                df = pd.read_csv(interactions_csv)
                interactions = list(zip(df['user_id'].tolist(), df['item_id'].tolist()))
                num_users = int(df['user_id'].max()) + 1
                num_items = int(df['item_id'].max()) + 1
                return RecommendationDataset(interactions, num_users, num_items)
            except Exception:
                pass

        interactions, n_users, n_items = DataPreprocessor.download_and_preprocess(dataset, dest_dir)
        return RecommendationDataset(interactions, n_users, n_items)
    
    @staticmethod
    def create_adversarial_edges(interactions: List[Tuple[int, int]], 
                               num_users: int, 
                               num_items: int,
                               num_adversarial: int = 100) -> List[Tuple[int, int]]:
        """Create adversarial edges by sampling least probable interactions"""
        # Simple implementation: sample random non-existing edges
        existing_edges = set(interactions)
        adversarial_edges = []
        
        while len(adversarial_edges) < num_adversarial:
            user = np.random.randint(0, num_users)
            item = np.random.randint(0, num_items)
            
            if (user, item) not in existing_edges and (user, item) not in adversarial_edges:
                adversarial_edges.append((user, item))
        
        return adversarial_edges
    
    @staticmethod
    def sample_negative_edges(interactions: List[Tuple[int, int]],
                            num_users: int,
                            num_items: int,
                            num_negative: int = 1000) -> List[Tuple[int, int]]:
        """Sample negative edges for evaluation"""
        existing_edges = set(interactions)
        negative_edges = []
        
        while len(negative_edges) < num_negative:
            user = np.random.randint(0, num_users)
            item = np.random.randint(0, num_items)
            
            if (user, item) not in existing_edges and (user, item) not in negative_edges:
                negative_edges.append((user, item))
        
        return negative_edges