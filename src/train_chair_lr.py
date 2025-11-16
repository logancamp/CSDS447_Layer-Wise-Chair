import argparse, joblib, json, time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_recall_fscore_support, classification_report,
    balanced_accuracy_score, make_scorer
)

# from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
"""
SUMMARY:
Train a logistic regression classifier to detect hallucinations based on features extracted by featurize.py.
Saves the trained model as a pickle file for later use in predict_chair.py.
"""

def main():
    # Add command-line args
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True) # CSV from featurize.py
    ap.add_argument("--out", default="outputs/chair_classifier_lr.pkl") # clf model output path
    args = ap.parse_args()
    
    # --- seeds ---
    MODEL_SEED = 1            # LR: 1 1b, 242 8b, 1 q4bi, 1 4bt, 1 q8bi, 1 m8b
    SPLIT_SEED = 23
    SMOTE_BASE = 42
    DOWNSAMPLE_SEED = 323       # NN | model: 323 1b, 7 8b, 283 q4bi, 451 4bt, 449 q8bi, 421 m8b

    # Load features
    """
    TODO: 
        abstract this reading step to a common util function and include the path to the layer data and the final layer cap, 
        in this function if final layer == "all" format the same, just import features like this, 
        otherwise cut off features at the cap layer and adjust the final layer logprobs and entropies from the 
        featurize_hist output. 
        Note: format the hist final layer the same as featurize.py file does for last layer
              Look at features for formatting, and featurize_hist for full layer hist formats:
                last_lp_tail_{i} and last_ent_tail_{i} per token
    """ 
    df = pd.read_csv(args.features)
    df = df.replace([np.inf, -np.inf], np.nan)

    # freeze exact feature order (drop meta)
    feature_cols = [c for c in df.columns if c not in ("qid","split","y")]

    # use baked-in splits
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise SystemExit(f"Empty split(s): train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    Xtr, ytr = train_df[feature_cols].values.tolist(), train_df["y"].astype(int).values.tolist()
    Xva, yva = val_df[feature_cols].values.tolist(), val_df["y"].astype(int).values.tolist()
    Xte, yte = test_df[feature_cols].values.tolist(), test_df["y"].astype(int).values.tolist()

    #############################################################
    #############################################################
    # Define pipeline with SMOTE and logistic regression
    """ pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("vth", VarianceThreshold(threshold=1e-6)),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("smote", SMOTE(random_state=SMOTE_BASE)),
        ("clf", LogisticRegression(
            penalty="elasticnet", 
            solver="saga", 
            l1_ratio=0.5,
            max_iter=20000,
            random_state=MODEL_SEED
        )),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED)
    scoring = make_scorer(average_precision_score, response_method="predict_proba")
    param_grid = {
        "smote": [SMOTE(random_state=SMOTE_BASE)],
        "smote__sampling_strategy": [0.5, 0.7, 0.9, 1.0],
        "smote__k_neighbors": [3, 5],
        "clf__C": np.logspace(-4, -1, 5).tolist(),
        "clf__l1_ratio": [0.5, 1.0],
        "clf__class_weight": [None, "balanced"],
    }

    gs = GridSearchCV(pipe, param_grid, cv=cv, scoring=scoring, n_jobs=-1, refit=True) """
    
    #############################################################
    # Train logistic regression with standard scaling
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("vth", VarianceThreshold(threshold=1e-6)),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegressionCV(
            Cs=np.logspace(-4, -1, 5).tolist(),
            cv=5,
            scoring="average_precision",
            penalty="elasticnet",
            solver="saga",
            l1_ratios=[0.5, 1.0],
            class_weight=None,
            max_iter=20000,
            n_jobs=-1,
            refit=True,
            random_state=MODEL_SEED
        ))
    ])
    ############################################################# 
    # Downsample majority class in TRAIN only
    """ from sklearn.utils import resample
    print(f"Skew test: {train_df.y.value_counts()}")
    maj = train_df[train_df.y == 1]
    minr = train_df[train_df.y == 0]

    if len(maj) > len(minr):
        maj_down = pd.DataFrame(resample(maj, replace=False, n_samples=len(minr), random_state=DOWNSAMPLE_SEED))
        train_df = pd.concat([maj_down, minr], ignore_index=True)
    else:
        minr_down = pd.DataFrame(resample(minr, replace=False, n_samples=len(maj), random_state=DOWNSAMPLE_SEED))
        train_df = pd.concat([maj, minr_down], ignore_index=True)
        
    Xtr = train_df[feature_cols].values.tolist()
    ytr = train_df["y"].astype(int).values.tolist()  """
    
    #############################################################
    # Set class weights manually (test on all)
    """ pipe.named_steps["clf"].class_weight = {0: len(ytr)/sum(np.array(ytr)==0), 1: len(ytr)/sum(np.array(ytr)==1)} """
    
    # for simple pipeline
    pipe.fit(Xtr, ytr)
    
    # for grid search pipeline with smote
    """ gs.fit(Xtr, ytr)
    pipe = gs.best_estimator_ """
    
    #############################################################
    #############################################################
    
    # Predict VAL and TEST probabilities
    proba_val = pipe.predict_proba(Xva)[:, 1]
    proba_te = pipe.predict_proba(Xte)[:, 1]

    # Threshold tuning on VAL (F1)
    ths = [i/100 for i in range(5, 96)]  # 0.05–0.95
    best_thr, best_bal = max(
        ((t, balanced_accuracy_score(yva, (proba_val >= t).astype(int))) for t in ths),
        key=lambda x: x[1]
    )
    print(f"[VAL] Best threshold by Balanced Accuracy: {best_thr:.2f} (BalAcc={best_bal:.3f})")

    # Apply same threshold to VAL and TEST
    yhat_val = (proba_val >= best_thr).astype(int)
    yhat_te = (proba_te >= best_thr).astype(int)

    # Metrics (VAL and TEST)
    val_auc = roc_auc_score(yva, proba_val); val_ap = average_precision_score(yva, proba_val)
    val_acc = accuracy_score(yva, yhat_val)
    val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(yva, yhat_val, average="binary", zero_division=0)

    te_auc = roc_auc_score(yte, proba_te); te_ap = average_precision_score(yte, proba_te)
    te_acc = accuracy_score(yte, yhat_te)
    te_prec, te_rec, te_f1, _ = precision_recall_fscore_support(yte, yhat_te, average="binary", zero_division=0)

    print(f"[VAL ] AUC={val_auc:.3f} | AP={val_ap:.3f} | ACC={val_acc:.3f} | F1={val_f1:.3f}")
    print(f"[TEST] AUC={te_auc:.3f} | AP={te_ap:.3f} | ACC={te_acc:.3f} | F1={te_f1:.3f}")
    print(classification_report(yte, yhat_te, digits=3))

    # Build metrics dict for saving
    metrics = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "splits": {"train_n": int(len(ytr)), "val_n": int(len(yva)), "test_n": int(len(yte))},
        "val":  {
            "auc_roc": float(val_auc), "avg_precision": float(val_ap),
            "accuracy": float(val_acc), "precision": float(val_prec),
            "recall": float(val_rec), "f1": float(val_f1)
        },
        "test": {
            "auc_roc": float(te_auc), "avg_precision": float(te_ap),
            "accuracy": float(te_acc), "precision": float(te_prec),
            "recall": float(te_rec), "f1": float(te_f1)
        },
        "feature_count": len(feature_cols),
        "feature_names": feature_cols,
        "threshold": float(best_thr)
    }

    # Save model
    joblib.dump(pipe, args.out)
    print(f"Saved model: {args.out}")
    
    # Save metrics
    metrics_path = args.out.replace(".pkl", ".train_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {metrics_path}")
    
    
if __name__ == "__main__":
    main()