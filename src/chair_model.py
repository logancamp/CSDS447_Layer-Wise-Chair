import torch
import torch.nn as nn
from torch.utils.data import Dataset # <-- Added import
import json # <-- Added import

"""
Defines the CHAIRModel, a hybrid attention network.
It processes:
1. A set of scalar features (from historical layers).
2. Two sequences (logprobs and entropies) from the last K tokens.
"""

class CHAIRModel(nn.Module):
    def __init__(self, n_scalar_features, k_seq_len, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.k_seq_len = k_seq_len

        # --- 1. Scalar Feature Processor ---
        # A simple MLP to project scalar features to d_model
        self.scalar_proj = nn.Sequential(
            nn.Linear(n_scalar_features, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )

        # --- 2. Sequence Feature Processor ---
        # Projects the two sequence features (lp, ent) into d_model
        self.sequence_proj = nn.Linear(2, d_model) # 2 features: lp and ent
        
        # A standard Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # A special [CLS] token, like in BERT, to aggregate sequence info
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # --- 3. Final Classification Head ---
        # Takes the concatenated [Scalar_Vec, Sequence_Vec]
        self.classification_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 1) # Final logit
        )

    def forward(self, scalar_feats, lp_seq, ent_seq):
        # 1. Process Scalars
        # scalar_feats shape: (Batch, N_Scalars)
        scalar_vec = self.scalar_proj(scalar_feats) # -> (Batch, d_model)

        # 2. Process Sequences
        # lp_seq, ent_seq shapes: (Batch, K)
        # Stack them to (Batch, K, 2)
        sequences = torch.stack([lp_seq, ent_seq], dim=-1)
        # Project to (Batch, K, d_model)
        sequence_vecs = self.sequence_proj(sequences)

        # 3. Prepend [CLS] token
        # cls_token shape: (1, 1, d_model) -> (Batch, 1, d_model)
        batch_size = scalar_feats.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # full_seq shape: (Batch, K + 1, d_model)
        full_seq = torch.cat([cls_tokens, sequence_vecs], dim=1)

        # 4. Pass through Transformer
        # transformer_out shape: (Batch, K + 1, d_model)
        transformer_out = self.transformer_encoder(full_seq)

        # 5. Get the aggregated sequence vector (the [CLS] token output)
        # sequence_pooled shape: (Batch, d_model)
        sequence_pooled = transformer_out[:, 0] # Get the first token

        # 6. Combine and Classify
        # combined_vec shape: (Batch, d_model * 2)
        combined_vec = torch.cat([scalar_vec, sequence_pooled], dim=1)
        
        # logit shape: (Batch, 1)
        logit = self.classification_head(combined_vec)
        return logit

# --- MOVED FROM train_chair_nn.py ---
# This code is now shared between training and prediction

class ChairDataset(Dataset):
    """
    PyTorch Dataset to load the .jsonl features created by featurize_nn.py.
    """
    def __init__(self, features_jsonl_path, scalar_feature_names):
        self.features = []
        self.scalar_feature_names = scalar_feature_names
        
        with open(features_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.features.append(json.loads(line))
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        record = self.features[idx]
        
        # 1. Get Label
        y = record['y']
        
        # 2. Get Scalar features in the correct order
        scalar_map = record['scalar_features']
        scalar_vec = [scalar_map.get(key, 0.0) for key in self.scalar_feature_names]
        
        # 3. Get Sequence features
        seq_feats = record['sequence_features']
        lp_seq = seq_feats['last_lp_tail_vec']
        ent_seq = seq_feats['last_ent_tail_vec']
        
        return (
            torch.tensor(scalar_vec, dtype=torch.float32),
            torch.tensor(lp_seq, dtype=torch.float32),
            torch.tensor(ent_seq, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )

def collate_fn(batch):
    """
    Custom collate function to stack batch tensors.
    """
    scalar_vecs, lp_seqs, ent_seqs, labels = zip(*batch)
    
    return (
        torch.stack(scalar_vecs),
        torch.stack(lp_seqs),
        torch.stack(ent_seqs),
        torch.stack(labels)
    )
