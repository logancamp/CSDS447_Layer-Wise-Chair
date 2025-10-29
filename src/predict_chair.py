import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    accuracy_score, f1_score, precision_score, recall_score
)
import time
import csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_pkl", required=True, help="trained classifier dir from train_chair.py")
    ap.add_argument("--train_metrics", required=True, help="metrics json dir from training data")
    ap.add_argument("--test_data", required=True, help="original JSONL test data dir used to generate features")
    ap.add_argument("--features", required=True, help="features CSV dir from featurize.py")
    args = ap.parse_args()

    # Load features produced by featurize.py (same schema used for training)
    df = pd.read_csv(args.features)
    feature_cols = [c for c in df.columns if c != "y"]
    X = df[feature_cols].values
    
    # --- Load tuned threshold from training (fallback to 0.5) ---
    thr = 0.5
    try:
        with open(args.train_metrics, "r") as f:
            train_meta = json.load(f)
        if isinstance(train_meta, dict) and "threshold" in train_meta:
            thr = float(train_meta["threshold"])
    except Exception:
        # Keep default 0.5 if file missing or malformed
        pass

    # Load model and score
    clf = joblib.load(args.model_pkl)
    prob = clf.predict_proba(X)[:, 1]  # P(hallucination)

    # Append evaluation metrics into the existing training metrics file
    if "y" in df.columns:
        y_true  = df["y"].to_numpy(dtype=int).reshape(-1)
        y_score = np.asarray(prob, dtype=float).reshape(-1)
        y_pred  = (y_score >= thr).astype(int)  # use learned threshold

        metrics = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_train": None,  # not known here, leave as None
            "n_test": int(len(df)),
            "threshold": float(thr),
            "auc_roc": float(roc_auc_score(y_true, y_score)),
            "avg_precision": float(average_precision_score(y_true, y_score)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred)),
        }

        # Save metrics for the model
        metrics_path = args.model_pkl.replace(".pkl", ".predict_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics: {metrics_path}")

    # Safely overwrite the original JSONL (write to tmp, then replace)
    src = Path(args.test_data)

    # sanity: check counts first (non-empty lines only)
    with open(src, "r", encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]
    assert len(prob) == len(lines), f"Mismatch: {len(prob)} probs vs {len(lines)} JSONL rows"

    tmp = src.with_suffix(".tmp.jsonl")
    with open(tmp, "w", encoding="utf-8") as f_out:
        for p, line in zip(prob, lines):
            r = json.loads(line)
            r["chair_score"] = float(p)
            r["chair_label"] = int(p >= thr)
            f_out.write(json.dumps(r, ensure_ascii=False) + "\n")

    tmp.replace(src)  # atomic replace of the original file
    print(f"Wrote (overwrote): {src}\nMean chair_score={float(np.mean(prob)):.3f}")
    
    # Write a CSV summary
    scores_csv = src.with_suffix(".chair_scores.csv")
    with open(src, "r", encoding="utf-8") as f_in, open(scores_csv, "w", newline="", encoding="utf-8") as f_out:
        w = csv.writer(f_out)
        w.writerow(["qid", "question", "true_answer", "predicted_answer", "correct", "predicted", "score"])
        for p, line in zip(prob, f_in):
            if not line.strip():
                continue
            r = json.loads(line)
            qid = r.get("qid")
            question = r.get("question", "")
            choices = r.get("choices", [])
            labels = r.get("labels", [])
            true_idx = labels.index(1) if 1 in labels else None
            true_answer = choices[true_idx] if (true_idx is not None and true_idx < len(choices)) else ""
            predicted_answer = r.get("pred_text", "")
            correct = bool(r.get("correct", False))
            predicted = int(p >= thr)
            w.writerow([qid, question, true_answer, predicted_answer, int(correct), predicted, float(p)])
    print(f"Wrote: {scores_csv}")

if __name__ == "__main__":
    main()