import argparse, json, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import pandas as pd
from torch.utils.data import TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from tqdm import tqdm
import numpy as np
import sys
from chair_model import CHAIRModel

"""
Trains the CHAIRModel neural network.
Reads the .jsonl features from featurize_nn.py, splits into train/val,
and trains the model, saving the best checkpoint.
"""

def fetch_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="Path to features .csv file from featurize.py")
    ap.add_argument("--out", default="outputs/chair_nn.pth", help="Path to save the trained model")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--d_model", type=int, default=128, help="Model embedding dimension")
    ap.add_argument("--nhead", type=int, default=4, help="Number of transformer attention heads")
    ap.add_argument("--num_layers", type=int, default=2, help="Number of transformer encoder layers")
    return ap.parse_args()


# --- THIS CODE WAS MOVED TO chair_model.py ---
# class ChairDataset(Dataset):
#    ...
# def collate_fn(batch):
#    ...
# ----------------------------------------------


def main():
    args = fetch_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Metadata from CSV ---
    cdf = pd.read_csv(args.features)
    if not {"y", "split"}.issubset(cdf.columns):
        raise SystemExit("CSV must contain columns: 'y' and 'split'.")
    feature_cols = [c for c in cdf.columns if c not in ("qid", "split", "y")]
    scalar_feature_names = feature_cols
    K = 1  # use 1-step dummy sequences to satisfy the model's seq inputs
    print(f"Loaded CSV. Scalar features: {len(scalar_feature_names)} | K={K}")

    # --- 2. Load Data and Create Splits (from CSV) ---
    print("Loading data...")
    X  = torch.tensor(cdf[scalar_feature_names].values, dtype=torch.float32)
    y  = torch.tensor(cdf["y"].astype(int).values, dtype=torch.float32)
    lp = torch.zeros((X.size(0), K), dtype=torch.float32)   # dummy logprob sequence
    ent= torch.zeros((X.size(0), K), dtype=torch.float32)   # dummy entropy sequence
    full_dataset = TensorDataset(X, lp, ent, y)

    splits_by_idx = cdf["split"].astype(str).tolist()
    labels_by_idx = cdf["y"].astype(int).tolist()

    train_indices = [i for i, s in enumerate(splits_by_idx) if s == "train"]
    val_indices   = [i for i, s in enumerate(splits_by_idx) if s == "val"]
    test_indices  = [i for i, s in enumerate(splits_by_idx) if s == "test"]  # optional

    if not train_indices or not val_indices:
        raise SystemExit(f"Empty split(s): train={len(train_indices)} val={len(val_indices)} test={len(test_indices)}")

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset   = Subset(full_dataset, val_indices)

    print(f"Data loaded: train={len(train_dataset)} | val={len(val_dataset)} | test={len(test_indices)}")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False
    )

    # --- 3. Initialize Model and Optimizer ---
    model = CHAIRModel(
        n_scalar_features=len(scalar_feature_names),
        k_seq_len=K,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Class weights from TRAIN only
    train_labels = [labels_by_idx[i] for i in train_indices]
    pos = sum(train_labels)
    neg = len(train_labels) - pos
    if pos == 0:
        raise SystemExit("No positive examples in train split; cannot compute pos_weight.")
    pos_weight = neg / pos
    print(f"Using positive class weight (train only): {pos_weight:.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    # --- 4. Training Loop ---
    best_val_auc = -1.0
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            scalar_feats, lp_seqs, ent_seqs, labels = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            logits = model(scalar_feats, lp_seqs, ent_seqs).squeeze(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # --- Validation Loop ---
        model.eval()
        total_val_loss = 0
        all_labels = []
        all_probs = []
        
        with torch.inference_mode():
            for batch in val_loader:
                scalar_feats, lp_seqs, ent_seqs, labels = [b.to(device) for b in batch]
                
                logits = model(scalar_feats, lp_seqs, ent_seqs).squeeze(-1)
                loss = criterion(logits, labels)
                total_val_loss += loss.item()
                
                probs = torch.sigmoid(logits)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        
        # Calculate validation metrics
        val_auc = roc_auc_score(all_labels, all_probs)
        val_ap = average_precision_score(all_labels, all_probs)
        val_f1 = f1_score(all_labels, (np.array(all_probs) >= 0.5).astype(int))

        print(f"Epoch {epoch+1:2}/{args.epochs} | Train Loss: {avg_train_loss:7.4f} | Val Loss: {avg_val_loss:7.4f} | "
              f"Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f} | Val F1: {val_f1:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            print(f"  -> New best model saved to {args.out} (AUC: {val_auc:.4f})")
            torch.save({
                'state_dict': model.state_dict(),
                'params': {
                    'n_scalar_features': len(scalar_feature_names),
                    'k_seq_len': K,
                    'd_model': args.d_model,
                    'nhead': args.nhead,
                    'num_layers': args.num_layers
                }
            }, args.out)

    print(f"\nTraining finished in {time.time() - start_time:.2f} seconds.")

    # --- 5. Final Evaluation on Validation Set ---
    print("Evaluating best model on validation set...")
    # Load best model
    best_model_state = torch.load(args.out, map_location=device)
    model.load_state_dict(best_model_state['state_dict'])
    model.eval()

    all_labels = []
    all_probs = []
    with torch.inference_mode():
        for batch in val_loader:
            scalar_feats, lp_seqs, ent_seqs, labels = [b.to(device) for b in batch]
            logits = model(scalar_feats, lp_seqs, ent_seqs).squeeze(-1)
            probs = torch.sigmoid(logits)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    val_auc = roc_auc_score(all_labels, all_probs)
    val_ap = average_precision_score(all_labels, all_probs)
    val_f1 = f1_score(all_labels, (np.array(all_probs) >= 0.5).astype(int))
    
    metrics = {
        "best_val_auc": val_auc,
        "best_val_ap": val_ap,
        "best_val_f1": val_f1,
        "n_train": len(train_dataset),
        "n_val": len(val_dataset),
        "params": best_model_state['params']
    }
    
    metrics_path = Path(args.out).with_suffix(".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved final validation metrics: {metrics_path}")


if __name__ == "__main__":
    main()
