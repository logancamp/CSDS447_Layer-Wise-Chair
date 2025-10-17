import argparse, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_recall_fscore_support, confusion_matrix, classification_report
)
import json, time

"""
SUMMARY:
Train a logistic regression classifier to detect hallucinations based on features extracted by featurize.py.
Saves the trained model as a pickle file for later use in predict_chair.py.
"""

def main():
    # Add command-line args
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True) # CSV from featurize.py
    ap.add_argument("--out", default="outputs/chair_classifier.pkl") # clf model output path
    args = ap.parse_args()

    # Load features
    df = pd.read_csv(args.features)
    y = df["y"].values
    X = df.drop(columns=["y"]).values

    # Train/test split: 80/20 split, stratified to preserves class balance in both sets
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y.tolist())

    # Train logistic regression with standard scaling
    # TODO: expand to an attention-based model like in the paper
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)), # center and scale features with z-score
        ("clf", LogisticRegression(max_iter=200, class_weight="balanced"))
    ])
    
    # Fit and evaluate
    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(Xte)[:,1]

    # Define threshold and predictions
    thr = 0.5
    yhat = (proba >= thr).astype(int)

    # Compute metrics
    auc = roc_auc_score(yte, proba)
    ap  = average_precision_score(yte, proba)
    acc = accuracy_score(yte, yhat)
    prec, rec, f1, _ = precision_recall_fscore_support(yte, yhat, average="binary", zero_division=0)
    cm = confusion_matrix(yte, yhat).tolist()

    print(f"AUC={auc:.3f} | AP={ap:.3f} | ACC={acc:.3f} | F1={f1:.3f}")
    print(classification_report(yte, yhat, digits=3))

    # Build metrics dict for saving
    metrics = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "threshold": thr,
        "auc_roc": float(auc),
        "avg_precision": float(ap),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm,
    }

    # Save model + metrics
    joblib.dump(pipe, args.out)
    print(f"Saved model: {args.out}")

    metrics_path = args.out.replace(".pkl", ".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {metrics_path}")

    # Save next to the model
    metrics_path = args.out.replace(".pkl", ".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {metrics_path}")

if __name__ == "__main__":
    main()