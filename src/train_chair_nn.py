import argparse, json, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import pandas as pd
from torch.utils.data import TensorDataset
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, accuracy_score, precision_score, recall_score
)
from tqdm import tqdm
import numpy as np
from chair_model import CHAIRModel

"""
Trains the CHAIRModel neural network.
Reads the .jsonl features from featurize_nn.py, splits into train/val,
and trains the model, saving the best checkpoint.
"""

def fetch_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="Path to features .csv file from featurize.py")
    ap.add_argument("--out", default="outputs/chair_classifier_nn.pth", help="Path to save the trained model")
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
    
    leaky_names = {"y","label","labels","correct","gold","target","answer","pred_text"}
    leaky = [c for c in feature_cols if c.lower() in leaky_names]
    assert not leaky, f"Leaky target columns in features: {leaky}"

    print(f"Loaded CSV. Scalar features: {len(scalar_feature_names)} | K={K}")

    print("Loading data...")
    X = torch.tensor(
        cdf[scalar_feature_names]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32, copy=False)
    )
    # Train on hallucination: 1 = hallucination, 0 = correct
    y_np = cdf["y"].astype("int64").to_numpy()
    y = torch.tensor(y_np, dtype=torch.float32)
    
    lp = torch.zeros((X.size(0), K), dtype=torch.float32)   # dummy logprob sequence
    ent = torch.zeros((X.size(0), K), dtype=torch.float32)   # dummy entropy sequence
    full_dataset = TensorDataset(X, lp, ent, y)

    splits_by_idx = cdf["split"].astype(str).tolist()
    labels_by_idx = y_np.tolist()

    train_indices = [i for i, s in enumerate(splits_by_idx) if s == "train"]
    val_indices = [i for i, s in enumerate(splits_by_idx) if s == "val"]
    test_indices = [i for i, s in enumerate(splits_by_idx) if s == "test"]  # optional
    
    assert set(train_indices).isdisjoint(val_indices), "Train/Val overlap!"
    assert set(train_indices).isdisjoint(test_indices), "Train/Test overlap!"
    assert set(val_indices).isdisjoint(test_indices), "Val/Test overlap!"

    if not train_indices or not val_indices:
        raise SystemExit(f"Empty split(s): train={len(train_indices)} val={len(val_indices)} test={len(test_indices)}")

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices) if len(test_indices) > 0 else None

    print(f"Data loaded: train={len(train_dataset)} | val={len(val_dataset)} | test={len(test_indices)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False) if test_dataset is not None else None

    # --- 3. Initialize Model and Optimizer ---
    model = CHAIRModel(
        n_scalar_features=len(scalar_feature_names),
        k_seq_len=K,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Class weights from TRAIN only (positives = hallucinations)
    train_labels = [labels_by_idx[i] for i in train_indices]
    pos = sum(train_labels)                 # count hallucinations (1s)
    neg = len(train_labels) - pos
    if pos == 0:
        raise SystemExit("No positive examples (hallucinations) in train split; cannot compute pos_weight.")
    pos_weight = neg / pos
    print(f"Using positive-class (hallucination) weight: {pos_weight:.2f}")
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
                
                p_hall = torch.sigmoid(logits)  # P(hallucination)
                all_labels.extend(labels.cpu().numpy().astype(int))  # labels already 1=hallucination
                all_probs.extend(p_hall.cpu().numpy())

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

    # --- 5. Final Evaluation on Validation (and Test) ---
    print("Evaluating best model on validation (and test if present)...")
    best_model_state = torch.load(args.out, map_location=device)
    model.load_state_dict(best_model_state['state_dict'])
    model.eval()

    # Collect validation probabilities
    yva_list, pva_list = [], []
    with torch.inference_mode():
        for batch in val_loader:
            scalar_feats, lp_seqs, ent_seqs, labels = [b.to(device) for b in batch]
            logits = model(scalar_feats, lp_seqs, ent_seqs).squeeze(-1)
            
            p_hall = torch.sigmoid(logits)  # P(hallucination)
            yva_list.extend(labels.cpu().numpy().astype(int))
            pva_list.extend(p_hall.cpu().numpy())

    yva_arr = np.asarray(yva_list, dtype=int)
    pva_arr = np.asarray(pva_list, dtype=float)

    # Choose threshold by max-F1 on validation
    from sklearn.metrics import roc_curve, precision_recall_curve

    # diagnostics
    print(f"[Diag] val hallucination_rate={yva_arr.mean():.3f}, "
        f"p(min/q1/med/q3/max)={np.min(pva_arr):.3f}/"
        f"{np.percentile(pva_arr,25):.3f}/{np.median(pva_arr):.3f}/"
        f"{np.percentile(pva_arr,75):.3f}/{np.max(pva_arr):.3f}")

    # 1) Youden's J
    fpr, tpr, thr = roc_curve(yva_arr, pva_arr)
    j = tpr - fpr
    best_thr = float(thr[int(np.argmax(j))])

    def one_class(thr, scores):
        yhat = (scores >= thr).astype(int)
        return (yhat.min() == yhat.max())

    # 2) Guard against one-class thresholds
    if one_class(best_thr, pva_arr):
        prec, rec, thr_pr = precision_recall_curve(yva_arr, pva_arr)
        dif = np.abs(prec - rec[:-1])
        best_thr = float(thr_pr[int(np.argmin(dif))])
        if one_class(best_thr, pva_arr):
            q = 1.0 - float(yva_arr.mean())  # prevalence-matching
            best_thr = float(np.quantile(pva_arr, q))

    print(f"[Thr] chosen={best_thr:.4f}")


    # Validation metrics at best_thr
    yhat_va = (pva_arr >= best_thr).astype(int)
    try:
        val_auc = float(roc_auc_score(yva_arr, pva_arr))
    except ValueError:
        val_auc = float("nan")
    try:
        val_ap = float(average_precision_score(yva_arr, pva_arr))
    except ValueError:
        val_ap = float("nan")
        
    val_acc = float(accuracy_score(yva_arr, yhat_va))
    val_prec = float(precision_score(yva_arr, yhat_va, zero_division=0))
    val_rec = float(recall_score(yva_arr, yhat_va, zero_division=0))
    val_f1 = float(f1_score(yva_arr, yhat_va, zero_division=0))

    # Test metrics (if present) using SAME threshold
    te_auc = te_ap = te_acc = te_prec = te_rec = te_f1 = float("nan")
    if test_loader is not None:
        yte_list, pte_list = [], []
        with torch.inference_mode():
            for batch in test_loader:
                scalar_feats, lp_seqs, ent_seqs, labels = [b.to(device) for b in batch]
                logits = model(scalar_feats, lp_seqs, ent_seqs).squeeze(-1)
                
                p_hall = torch.sigmoid(logits)
                yte_list.extend(labels.cpu().numpy().astype(int))
                pte_list.extend(p_hall.cpu().numpy())
                
        yte_arr = np.asarray(yte_list, dtype=int)
        pte_arr = np.asarray(pte_list, dtype=float)
        yhat_te = (pte_arr >= best_thr).astype(int)
        try:
            te_auc = float(roc_auc_score(yte_arr, pte_arr))
        except ValueError:
            te_auc = float("nan")
        try:
            te_ap = float(average_precision_score(yte_arr, pte_arr))
        except ValueError:
            te_ap = float("nan")
        te_acc = float(accuracy_score(yte_arr, yhat_te))
        te_prec = float(precision_score(yte_arr, yhat_te, zero_division=0))
        te_rec = float(recall_score(yte_arr, yhat_te, zero_division=0))
        te_f1 = float(f1_score(yte_arr, yhat_te, zero_division=0))

    # Persist metrics in requested schema
    metrics = { 
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "splits": {
            "train_n": int(len(train_dataset)),
            "val_n": int(len(val_dataset)),
            "test_n": int(len(test_indices))
        },
        "val":  {
            "auc_roc": val_auc, 
            "avg_precision": val_ap,
            "accuracy": val_acc, 
            "precision": val_prec,
            "recall": val_rec, 
            "f1": val_f1
        },
        "test": {
            "auc_roc": te_auc, 
            "avg_precision": te_ap,
            "accuracy": te_acc, 
            "precision": te_prec,
            "recall": te_rec, 
            "f1": te_f1
        },
        "feature_count": int(len(scalar_feature_names)),
        "feature_names": list(map(str, scalar_feature_names)),
        "threshold": float(best_thr),
        "model_params": best_model_state['params']
    }

    metrics_path = Path(args.out).with_suffix(".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {metrics_path}")

if __name__ == "__main__":
    main()
