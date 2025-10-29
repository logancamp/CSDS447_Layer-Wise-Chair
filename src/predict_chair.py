import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_pkl", required=True)
    ap.add_argument("--features", required=True)
    args = ap.parse_args()

    # Load features produced by featurize.py (same schema used for training)
    df = pd.read_csv(args.features)
    X = df.drop(columns=["y"]).values

    # Load model and score
    clf = joblib.load(args.model_pkl)
    prob = clf.predict_proba(X)[:, 1]  # P(hallucination)

    # Add evaluation Metrics to json
    if "y" in df.columns:
        y_true  = df["y"].to_numpy(dtype=int).reshape(-1)
        y_score = np.asarray(prob, dtype=float).reshape(-1)

        y_pred = (y_score >= 0.5).astype(int) # threshold for class preds

        metrics = {
            "auc": float(roc_auc_score(y_true, y_score)),
            "avg_precision": float(average_precision_score(y_true, y_score)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred)),
        }

        metrics_path = Path(args.features).with_suffix(".metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved evaluation metrics → {metrics_path}")

    # Add chair_score back onto the original JSON
    src = Path(args.preds_jsonl)
    out = src.with_suffix(".chair_scored.jsonl")
    with open(src, "r", encoding="utf-8") as f_in, open(out, "w", encoding="utf-8") as f_out:
        for p, line in zip(prob, f_in):
            if not line.strip():
                continue
            r = json.loads(line)
            r["chair_score"] = float(p)
            f_out.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote: {out}\nMean chair_score={float(np.mean(prob)):.3f}")

if __name__ == "__main__":
    main()