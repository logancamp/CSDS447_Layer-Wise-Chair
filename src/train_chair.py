import argparse, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_recall_fscore_support, classification_report, f1_score
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegressionCV
import json, time
import numpy as np

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
    df = df.replace([np.inf, -np.inf], np.nan)  # new line: replace infinities
    y = df["y"].astype(int).values
    X = df.drop(columns=["y"]).values

    # Train/test split: 80/20 split, stratified to preserves class balance in both sets
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y.tolist())

    # Train logistic regression with standard scaling
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("vth", VarianceThreshold(threshold=1e-6)),
        ("clf", LogisticRegressionCV(
            Cs=np.logspace(-3,2,10).tolist(), 
            cv=5, scoring="roc_auc",
            penalty="elasticnet", 
            solver="saga", 
            l1_ratios=[0.0, 0.5, 1.0],
            class_weight="balanced", 
            max_iter=5000, n_jobs=-1, 
            refit=True
        ))
    ])
    
    # Fit and evaluate
    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(Xte)[:, 1]

    # Threshold tuning
    ths = [i / 100 for i in range(5, 96)]  # 0.05–0.95
    best_f1, best_thr = max(
        (f1_score(yte, (proba >= t).astype(int)), t) for t in ths
    )
    print(f"Best threshold based on F1: {best_thr:.2f} (F1={best_f1:.3f})")

    thr = best_thr
    yhat = (proba >= thr).astype(int)

    # Compute metrics
    auc = roc_auc_score(yte, proba)
    ap_score = average_precision_score(yte, proba)
    acc = accuracy_score(yte, yhat)
    prec, rec, f1, _ = precision_recall_fscore_support(yte, yhat, average="binary", zero_division=0)

    print(f"AUC={auc:.3f} | AP={ap_score:.3f} | ACC={acc:.3f} | F1={f1:.3f}")
    print(classification_report(yte, yhat, digits=3))

    # Build metrics dict for saving
    metrics = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "threshold": thr,
        "auc_roc": float(auc),
        "avg_precision": float(ap_score),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1)
    }

    # Save model
    joblib.dump(pipe, args.out)
    print(f"Saved model: {args.out}")

    # Save metrics for the model
    metrics_path = args.out.replace(".pkl", ".train_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {metrics_path}")

if __name__ == "__main__":
    main()