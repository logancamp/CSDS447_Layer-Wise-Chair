import argparse, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

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
    auc = roc_auc_score(yte, proba)
    ap = average_precision_score(yte, proba)
    
    # AUC = area under ROC curve, AP = average precision (area under precision-recall curve)
    print(f"AUC={auc:.3f} | AP={ap:.3f}")
    # return precision, recall, f1 for each class
    print(classification_report(yte, (proba>=0.5).astype(int), digits=3))

    # Save the trained model
    joblib.dump(pipe, args.out)
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
