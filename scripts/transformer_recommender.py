import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import json
from pathlib import Path
import pandas as pd

'''
Transformer-based movie recommender 
trained on INSPIRED dataset
'''
class TransformerRecommender(nn.Module):

    def __init__(self, model_name="bert-base-uncased", num_movies=20):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        
        # Recommendation head
        self.recommender_head = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_movies)
        )
        
    def forward(self, input_ids, attention_mask):
        # Get encoder outputs
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Get movie scores
        logits = self.recommender_head(cls_output)
        return logits
    
    def predict_top_k(self, conversation_text, movie_id_to_name, k=3):
        """Predict top-k movie recommendations"""
        # Tokenize
        inputs = self.tokenizer(
            conversation_text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Get predictions
        with torch.no_grad():
            logits = self.forward(inputs['input_ids'], inputs['attention_mask'])
            scores = torch.softmax(logits, dim=-1)
        
        # SAFETY CHECK: Adjust k to available size
        num_available = min(len(movie_id_to_name), scores.shape[1])
        actual_k = min(k, num_available)
        
        # Get top-k movies
        top_k_scores, top_k_indices = torch.topk(scores[0], actual_k)
        
        recommendations = []
        for idx, score in zip(top_k_indices, top_k_scores):
            movie_id = idx.item()
            if movie_id in movie_id_to_name:
                recommendations.append({
                    'movie_id': movie_id,
                    'movie_name': movie_id_to_name[movie_id],
                    'score': score.item()
                })
        
        return recommendations

'''
Process the INSPIRED dataset 
for training the transformer
'''
class INSPIREDDataProcessor:
    
    def __init__(self, dataset_dir="data"):
        self.dataset_dir = Path(dataset_dir)
        self.movie_id_map = {}
        self.movie_name_map = {}
    
    '''
    Load movie database and create mappings
    '''
    def load_movie_database(self):
        
        movie_db_path = self.dataset_dir / "raw" / "movie_database.tsv"
        
        if not movie_db_path.exists():
            raise FileNotFoundError(f"Movie database not found at {movie_db_path}")
        
        # Load with pandas
        df = pd.read_csv(movie_db_path, sep='\t', encoding='utf-8')
        
        print(f"Loading {len(df)} movies from database...")
        
        movie_name_col = 'title'
        
        # Create movie ID mappings
        for idx, row in df.iterrows():
            movie_name = str(row[movie_name_col])
            if pd.notna(movie_name) and movie_name != 'nan':
                self.movie_id_map[movie_name] = idx
                self.movie_name_map[idx] = movie_name
        
        print(f"Loaded {len(self.movie_id_map)} movies")
        return self.movie_id_map, self.movie_name_map  
        
    '''
    Load dialog data grouped by conversation
    '''
    def load_dialogs(self, split="train", max_dialogs=None):
        
        dialog_path = self.dataset_dir / "processed" / f"{split}.tsv"
        
        if not dialog_path.exists():
            raise FileNotFoundError(f"Dialog file not found at {dialog_path}")
        
        df = pd.read_csv(dialog_path, sep='\t')
        
        # Group by dialog_id
        dialogs = []
        for dialog_id, group in df.groupby('dialog_id'):
            conversation = []
            recommended_movies = []
            
            for _, row in group.iterrows():
                speaker = row.get('speaker', '')
                utterance = row.get('utterance', '')
                movie_name = row.get('movie_name', '')
                
                conversation.append(f"{speaker}: {utterance}")
                
                # Track recommended movies
                if movie_name and movie_name in self.movie_id_map:
                    recommended_movies.append(self.movie_id_map[movie_name])
            
            if conversation and recommended_movies:
                dialogs.append({
                    'dialog_id': dialog_id,
                    'conversation': ' '.join(conversation),
                    'recommended_movies': list(set(recommended_movies))
                })
            
            if max_dialogs and len(dialogs) >= max_dialogs:
                break
        
        print(f"Loaded {len(dialogs)} dialogs")
        return dialogs