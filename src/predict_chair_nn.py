import csv
import argparse, json, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader, SequentialSampler
import numpy as np
from tqdm import tqdm
import sys
from chair_model import CHAIRModel
import pandas as pd
from torch.utils.data import TensorDataset
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, f1_score
)

def fetch_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_pth", required=True, help="Path to trained chair_nn.pth model")
    ap.add_argument("--test_data", required=True, help="Path to the raw mc1_results.jsonl from eval_mc1")
    ap.add_argument("--features", required=True, help="Path to features CSV (mc1_results.features.csv)")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()
    return args

def main():
    args = fetch_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load metadata from CSV ---
    cdf = pd.read_csv(args.features)
    if not {"y", "split"}.issubset(cdf.columns):
        raise SystemExit("CSV must contain columns: 'y' and 'split'.")
    feature_cols = [c for c in cdf.columns if c not in ("qid","split","y")]
    scalar_feature_names = feature_cols
    K = 1 
    
    # --- Load Model ---
    try:
        model_state = torch.load(args.model_pth, map_location=device)
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
        print(f"Model loaded from {args.model_pth}")
    
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # --- Load Prediction Data from CSV ---
    X = torch.tensor(
        cdf[scalar_feature_names]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32, copy=False)
    )
    y = torch.tensor(cdf["y"].astype(int).values, dtype=torch.float32)
    lp = torch.zeros((X.size(0), K), dtype=torch.float32)   # dummy logprob seq
    ent = torch.zeros((X.size(0), K), dtype=torch.float32)   # dummy entropy seq
    dataset = TensorDataset(X, lp, ent, y)

    pred_loader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),  # keep order to match preds_jsonl
        batch_size=args.batch_size
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
    
    metrics_path = args.model_pth.replace(".pth", ".predict_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved evaluation metrics to: {metrics_path}")
    print(json.dumps(metrics, indent=2))

    # Write a CSV summary. Use df rows; enrich from JSONL if provided.
    if args.test_data:
        src = Path(args.test_data)
        with open(src, "r", encoding="utf-8") as f:
            recs = [json.loads(ln) for ln in f if ln.strip()]
        # Map by qid for fast lookup; prefer only split=='test' if present.
        by_qid = {}
        for r in recs:
            s = r.get("split")
            if (s is None) or (s == "test"):
                qid = r.get("qid")
                if qid is not None:
                    by_qid[qid] = r
    else:
        by_qid = None

    thr = 0.5  # Fixed threshold for binary classification - temp
    scores_csv = args.features.replace(".features", ".nn_chair_scores.csv")
    with open(scores_csv, "w", newline="", encoding="utf-8") as f_out:
        w = csv.writer(f_out)
        w.writerow(["qid","predicted","score","question","true_answer","predicted_answer","correct"])
        for (idx, row), p in zip(cdf.reset_index(drop=True).iterrows(), all_scores):
            qid = row.get("qid")
            predicted = int(p >= thr)
            question = true_answer = predicted_answer = ""
            correct = ""
            if by_qid and qid in by_qid:
                r = by_qid[qid]
                question = r.get("question", "")
                choices = r.get("choices", [])
                labels = r.get("labels", [])
                true_idx = labels.index(1) if 1 in labels else None
                true_answer = choices[true_idx] if (true_idx is not None and true_idx < len(choices)) else ""
                predicted_answer = r.get("pred_text", "")
                correct = int(bool(r.get("correct", False)))
            w.writerow([qid, predicted, float(p), question, true_answer, predicted_answer, correct])

    print(f"Wrote: {scores_csv}")
    print(f"Mean chair_score={float(np.mean(all_scores)):.3f}")

if __name__ == "__main__":
    main()

