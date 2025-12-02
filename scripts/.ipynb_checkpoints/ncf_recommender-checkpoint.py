"""
Neural Collaborative Filtering (NCF) Recommender for INSPIRED Dataset

FIXED: Proper index remapping to ensure contiguous indices 0 to N-1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
import re
from collections import defaultdict
from tqdm import tqdm


class NCFModel(nn.Module):
    """Neural Collaborative Filtering Model (NeuMF)"""
    
    def __init__(self, num_users, num_items, embedding_dim=64, mlp_layers=[128, 64, 32]):
        super(NCFModel, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        
        # GMF part - element-wise product
        self.gmf_user_embedding = nn.Embedding(num_users, embedding_dim)
        self.gmf_item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # MLP part - neural network
        self.mlp_user_embedding = nn.Embedding(num_users, embedding_dim)
        self.mlp_item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # MLP layers
        mlp_modules = []
        input_size = embedding_dim * 2
        for layer_size in mlp_layers:
            mlp_modules.append(nn.Linear(input_size, layer_size))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(0.2))
            input_size = layer_size
        self.mlp_layers = nn.Sequential(*mlp_modules)
        
        # Final prediction layer
        self.predict_layer = nn.Linear(embedding_dim + mlp_layers[-1], 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        nn.init.normal_(self.gmf_user_embedding.weight, std=0.01)
        nn.init.normal_(self.gmf_item_embedding.weight, std=0.01)
        nn.init.normal_(self.mlp_user_embedding.weight, std=0.01)
        nn.init.normal_(self.mlp_item_embedding.weight, std=0.01)
        
        for m in self.mlp_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        
        nn.init.xavier_uniform_(self.predict_layer.weight)
        nn.init.constant_(self.predict_layer.bias, 0)
    
    def forward(self, user_indices, item_indices):
        """Forward pass"""
        # GMF path
        gmf_user_embed = self.gmf_user_embedding(user_indices)
        gmf_item_embed = self.gmf_item_embedding(item_indices)
        gmf_output = gmf_user_embed * gmf_item_embed
        
        # MLP path
        mlp_user_embed = self.mlp_user_embedding(user_indices)
        mlp_item_embed = self.mlp_item_embedding(item_indices)
        mlp_input = torch.cat([mlp_user_embed, mlp_item_embed], dim=-1)
        mlp_output = self.mlp_layers(mlp_input)
        
        # Combine GMF and MLP
        combined = torch.cat([gmf_output, mlp_output], dim=-1)
        prediction = self.predict_layer(combined)
        
        return prediction.squeeze()
    
    def predict_top_k(self, user_idx, candidate_items, k=10):
        """Predict top-k items for a user"""
        self.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_idx] * len(candidate_items))
            
            if candidate_items.device != user_tensor.device:
                user_tensor = user_tensor.to(candidate_items.device)
            
            scores = self.forward(user_tensor, candidate_items)
            
            top_k_scores, top_k_indices = torch.topk(scores, k=min(k, len(scores)))
            
            recommendations = []
            for idx, score in zip(top_k_indices, top_k_scores):
                recommendations.append({
                    'item_idx': candidate_items[idx].item(),
                    'score': score.item()
                })
            
            return recommendations


class INSPIREDDataProcessor:
    """Process INSPIRED dataset for NCF training"""
    
    def __init__(self, dataset_dir="data"):
        self.dataset_dir = Path(dataset_dir)
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.movie_to_idx = {}
        self.idx_to_movie = {}
        self.interactions = []
    
    @staticmethod
    def strip_year(movie_name):
        """Remove (year) from movie name"""
        if pd.isna(movie_name):
            return None
        movie_name = str(movie_name).strip()
        movie_name = re.sub(r'\s*\(\d{4}\)\s*$', '', movie_name)
        return movie_name.strip()
    
    def load_movie_database(self):
        """Load movie database"""
        movie_db_path = self.dataset_dir / "raw" / "movie_database.tsv"
        
        if not movie_db_path.exists():
            raise FileNotFoundError(f"Movie database not found at {movie_db_path}")
        
        df = pd.read_csv(movie_db_path, sep='\t', encoding='utf-8')
        
        print(f"Loading {len(df)} movies from database...")
        
        movie_idx = 0
        for _, row in df.iterrows():
            movie_name = row.get('title', 'Unknown')
            
            if pd.isna(movie_name) or str(movie_name) == 'nan':
                continue
            
            self.movie_to_idx[movie_name] = movie_idx
            self.idx_to_movie[movie_idx] = movie_name
            movie_idx += 1
        
        print(f"Loaded {len(self.movie_to_idx)} movies")
        return df
    
    def match_movie_name(self, movie_name):
        """Match movie name to database"""
        if not movie_name or pd.isna(movie_name):
            return None
        
        # Strategy 1: Exact match
        if movie_name in self.movie_to_idx:
            return self.movie_to_idx[movie_name]
        
        # Strategy 2: Strip year and try again
        movie_no_year = self.strip_year(movie_name)
        
        for db_movie, idx in self.movie_to_idx.items():
            db_no_year = self.strip_year(db_movie)
            
            if movie_no_year and db_no_year:
                if movie_no_year.lower() == db_no_year.lower():
                    return idx
        
        return None
    
    def load_dialogs(self, split="train", max_dialogs=None):
        """Load dialogs and extract user-movie interactions"""
        dialog_path = self.dataset_dir / "processed" / f"{split}.tsv"
        
        if not dialog_path.exists():
            raise FileNotFoundError(f"Dialog file not found at {dialog_path}")
        
        df = pd.read_csv(dialog_path, sep='\t', encoding='utf-8')
        
        if max_dialogs:
            df = df.head(max_dialogs)
        
        print(f"\n{'='*60}")
        print(f"Loading interactions from {split}.tsv")
        print(f"{'='*60}")
        
        matched_count = 0
        unmatched_count = 0
        
        grouped = df.groupby('dialog_id')
        
        for dialog_id, dialog_df in tqdm(grouped, desc=f"Processing {split}"):
            if dialog_id not in self.user_to_idx:
                user_idx = len(self.user_to_idx)
                self.user_to_idx[dialog_id] = user_idx
                self.idx_to_user[user_idx] = dialog_id
            else:
                user_idx = self.user_to_idx[dialog_id]
            
            for _, row in dialog_df.iterrows():
                utterance = row.get('text', '')
                movies_str = row.get('movies', '')
                
                if pd.isna(movies_str) or not movies_str:
                    continue
                
                movie_names = [m.strip() for m in str(movies_str).split(';') if m.strip()]
                
                for movie_name in movie_names:
                    movie_idx = self.match_movie_name(movie_name)
                    
                    if movie_idx is not None:
                        rating = self._infer_rating(utterance)
                        self.interactions.append((user_idx, movie_idx, rating))
                        matched_count += 1
                    else:
                        unmatched_count += 1
        
        print(f"\nInteraction Summary:")
        print(f"  Matched: {matched_count}")
        print(f"  Unmatched: {unmatched_count}")
        print(f"  Match rate: {matched_count/(matched_count+unmatched_count)*100:.1f}%")
        print(f"{'='*60}\n")
        
        return len(self.interactions)
    
    def _infer_rating(self, utterance):
        """Infer rating from utterance sentiment"""
        utterance_lower = utterance.lower()
        
        if any(word in utterance_lower for word in ['love', 'favorite', 'amazing', 'excellent', 'best', 'masterpiece']):
            return 5.0
        elif any(word in utterance_lower for word in ['like', 'enjoy', 'good', 'great', 'recommend']):
            return 4.0
        elif any(word in utterance_lower for word in ['hate', 'terrible', 'worst', 'awful', 'horrible']):
            return 1.0
        elif any(word in utterance_lower for word in ['dislike', 'bad', 'boring', "didn't like"]):
            return 2.0
        else:
            return 3.0
    
    def get_train_data(self):
        """Get training data as arrays"""
        if not self.interactions:
            raise ValueError("No interactions loaded. Call load_dialogs first.")
        
        users = np.array([x[0] for x in self.interactions])
        items = np.array([x[1] for x in self.interactions])
        ratings = np.array([x[2] for x in self.interactions])
        
        return users, items, ratings
    
    def get_num_users(self):
        return len(self.user_to_idx)
    
    def get_num_items(self):
        return len(self.movie_to_idx)


def train_ncf(model, data_processor, num_epochs=10, batch_size=256, lr=0.001, 
              negative_samples=4, device='cpu'):
    """
    Train NCF model with proper index handling
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    # Get training data
    users, items, ratings = data_processor.get_train_data()
    
    # CRITICAL: Get unique items that actually exist in interactions
    unique_items_in_data = set(items)
    all_items_list = list(unique_items_in_data)
    
    # Create user-item interaction set for negative sampling
    user_item_set = set(zip(users, items))
    
    # Track history
    history = {'train_loss': [], 'epoch': []}
    
    print("="*60)
    print("TRAINING NCF MODEL")
    print("="*60)
    print(f"Total interactions: {len(users)}")
    print(f"Users: {data_processor.get_num_users()}")
    print(f"Items: {data_processor.get_num_items()}")
    print(f"Unique items in training: {len(unique_items_in_data)}")
    print(f"Negative samples per positive: {negative_samples}")
    print("="*60)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Shuffle data
        indices = np.random.permutation(len(users))
        
        # Create batches
        for start_idx in range(0, len(users), batch_size):
            end_idx = min(start_idx + batch_size, len(users))
            batch_indices = indices[start_idx:end_idx]
            
            batch_users = users[batch_indices]
            batch_items = items[batch_indices]
            batch_ratings = ratings[batch_indices]
            
            # Generate negative samples
            neg_items_list = []
            for user_idx, pos_item in zip(batch_users, batch_items):
                # Get items user hasn't interacted with (from actual items in data)
                user_items = {item for u, item in user_item_set if u == user_idx}
                neg_candidates = list(unique_items_in_data - user_items)
                
                if len(neg_candidates) < negative_samples:
                    # If not enough, repeat
                    neg_samples = neg_candidates * (negative_samples // len(neg_candidates) + 1)
                    neg_samples = neg_samples[:negative_samples]
                else:
                    neg_samples = np.random.choice(neg_candidates, negative_samples, replace=False)
                
                neg_items_list.extend(neg_samples)
            
            # Combine positive and negative samples
            all_users = np.concatenate([batch_users] * (negative_samples + 1))
            all_items_batch = np.concatenate([batch_items, neg_items_list])
            all_labels = np.concatenate([np.ones(len(batch_users)), 
                                        np.zeros(len(neg_items_list))])
            
            # Convert to tensors
            user_tensor = torch.LongTensor(all_users).to(device)
            item_tensor = torch.LongTensor(all_items_batch).to(device)
            label_tensor = torch.FloatTensor(all_labels).to(device)
            
            # Forward pass
            predictions = model(user_tensor, item_tensor)
            loss = criterion(predictions, label_tensor)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        history['train_loss'].append(avg_loss)
        history['epoch'].append(epoch + 1)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
    
    print("="*60)
    print("Training complete!")
    
    return model, history