# src/train_chair.py
import argparse, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)  # CSV from featurize.py
    ap.add_argument("--out", default="outputs/chair_clf.pkl")
    args = ap.parse_args()

    df = pd.read_csv(args.features)
    y = df["y"].values
    X = df.drop(columns=["y"]).values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(max_iter=200, class_weight="balanced"))
    ])
    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(Xte)[:,1]
    auc  = roc_auc_score(yte, proba)
    ap   = average_precision_score(yte, proba)
    print(f"AUC={auc:.3f} | AP={ap:.3f}")
    print(classification_report(yte, (proba>=0.5).astype(int), digits=3))

    joblib.dump(pipe, args.out)
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
