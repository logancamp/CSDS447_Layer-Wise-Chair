import argparse, json, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader, SequentialSampler
import numpy as np
from chair_model import CHAIRModel, ChairDataset, collate_fn
from tqdm import tqdm
import sys
# --- MISSING IMPORTS ADDED HERE ---
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, f1_score
)
# ----------------------------------


def fetch_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="Path to trained chair_nn.pth model")
    ap.add_argument("--preds_jsonl", required=True, help="Path to the raw eval_run.jsonl from eval_mc1")
    ap.add_argument("--features_jsonl", required=True, help="Path to the featurize_nn.py output (eval_run.features.jsonl)")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()
    return args

def main():
    args = fetch_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Metadata ---
    meta_path = Path(args.features_jsonl).with_suffix(".meta.json")
    if not meta_path.exists():
        print(f"Error: Could not find metadata file at {meta_path}")
        print("Please ensure it exists (it should be created by featurize_nn.py)")
        sys.exit(1)
        
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    scalar_feature_names = meta["scalar_feature_names"]
    K = meta["K"]
    
    # --- Load Model ---
    try:
        model_state = torch.load(args.model_path, map_location=device)
        model_params = model_state['params']
        
        print(f"Loaded model params: d_model={model_params['d_model']}, nhead={model_params['nhead']}, num_layers={model_params['num_layers']}")

        model = CHAIRModel(
            n_scalar_features=len(scalar_feature_names),
            k_seq_len=K,
            d_model=model_params.get('d_model', 128),
            nhead=model_params.get('nhead', 4),
            num_layers=model_params.get('num_layers', 2)
        )
        model.load_state_dict(model_state['state_dict'])
        model.to(device)
        model.eval()
        print(f"Model loaded from {args.model_path}")
    
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # --- Load Prediction Data ---
    dataset = ChairDataset(
        features_jsonl_path=args.features_jsonl,
        scalar_feature_names=scalar_feature_names
    )
    
    pred_loader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset), # Keep order for matching
        batch_size=args.batch_size,
        collate_fn=collate_fn
    )

    # --- Run Predictions ---
    all_scores = []
    y_true = []
    
    with torch.inference_mode():
        for batch in tqdm(pred_loader, desc="Predicting"):
            scalar_feats, lp_seqs, ent_seqs, labels = [b.to(device) for b in batch]
            
            logits = model(scalar_feats, lp_seqs, ent_seqs)
            probs = torch.sigmoid(logits.squeeze(-1)) # Convert logits to P(hallucination)
            
            all_scores.extend(probs.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    y_score = np.array(all_scores)
    y_true = np.array(y_true)
    y_pred = (y_score >= 0.5).astype(int) # Get binary predictions

    # --- Calculate and Save Metrics ---
    metrics = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_test": len(y_true),
        "auc": float(roc_auc_score(y_true, y_score)),
        "avg_precision": float(average_precision_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }
    
    metrics_path = Path(args.features_jsonl).with_suffix(".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved evaluation metrics to: {metrics_path}")
    print(json.dumps(metrics, indent=2))

    # --- Write Scores back to original JSONL ---
    src = Path(args.preds_jsonl)
    out = src.with_suffix(".chair_scored.jsonl")
    
    with open(src, "r", encoding="utf-8") as f_in, open(out, "w", encoding="utf-8") as f_out:
        for p, line in zip(all_scores, f_in):
            if not line.strip():
                continue
            r = json.loads(line)
            r["chair_score"] = float(p) # Add the new NN-based score
            f_out.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote scored predictions to: {out}")
    print(f"Mean chair_score={float(np.mean(all_scores)):.3f}")


if __name__ == "__main__":
    main()

