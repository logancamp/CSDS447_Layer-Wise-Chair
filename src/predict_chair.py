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
    ap.add_argument("--test_data", required=True, help="original JSONL test data used to enrich CSV summary")
    ap.add_argument("--features", required=True, help="features CSV dir from featurize.py")
    args = ap.parse_args()

    # Load features produced by featurize.py
    df = pd.read_csv(args.features)

    # If split column exists, use only the test rows
    if "split" in df.columns:
        df = df[df["split"] == "test"].copy()
        if df.empty:
            raise SystemExit("No rows in features with split=='test'. Check your featurization/split labels.")

    # Load threshold and expected feature order from training metrics
    with open(args.train_metrics, "r") as f:
        train_meta = json.load(f)

    thr = float(train_meta.get("threshold", 0.5))
    exp_cols = train_meta.get("feature_names")
    if not exp_cols:
        raise SystemExit("Training metrics missing 'feature_names'. Re-run training to store the schema.")

    # Drop meta columns initially
    feature_cols = [c for c in df.columns if c not in ("qid", "split", "y")]

    # Enforce same feature order if available
    if exp_cols:
        for c in exp_cols:
            if c not in df.columns:
                df[c] = 0.0
        if {"qid", "split", "y"}.issubset(df.columns):
            df = df.reindex(columns=["qid", "split", "y"] + list(exp_cols))
        else:
            df = df.reindex(columns=list(exp_cols))
        feature_cols = list(exp_cols)

    X = df[feature_cols].values

    # Load model and score
    clf = joblib.load(args.model_pkl)
    prob = clf.predict_proba(X)[:, 1]  # P(hallucination)

    # Append evaluation metrics into the existing training metrics file
    if "y" in df.columns:
        y_true = df["y"].astype(int).to_numpy().reshape(-1)
        y_score = np.asarray(prob, dtype=float).reshape(-1)
        y_pred = (y_score >= thr).astype(int)  # use learned threshold

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

    scores_csv = Path(args.features).with_suffix(".chair_scores.csv")
    with open(scores_csv, "w", newline="", encoding="utf-8") as f_out:
        w = csv.writer(f_out)
        w.writerow(["qid","predicted","score","question","true_answer","predicted_answer","correct"])
        for (idx, row), p in zip(df.reset_index(drop=True).iterrows(), prob):
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
    print(f"Mean chair_score={float(np.mean(prob)):.3f}")

if __name__ == "__main__":
    main()