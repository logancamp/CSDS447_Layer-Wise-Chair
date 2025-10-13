# src/predict_chair.py
import argparse, json
from pathlib import Path
import joblib
import pandas as pd

def build_features_from_jsonl(path, K=32):
    import csv
    import math
    rows = []
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            y = int(r.get("hallucination", not r.get("correct", False)))
            def get(k, d=0.0): return r.get(k, d)
            scalars = [
                get("avg_logprob"), get("avg_prob"), get("perplexity"),
                get("lp_mean"), get("lp_std"), get("lp_min"), get("lp_max"),
                get("lp_first"), get("lp_last"), get("lp_delta"),
                get("ent_mean"), get("ent_std"), get("ent_min"), get("ent_max"),
                get("ent_first"), get("ent_last"), get("ent_delta"),
            ]
            seq = r.get("token_logprobs", [])
            seq = list(seq)[-K:]
            if len(seq) < K: seq = [0.0]*(K-len(seq)) + seq
            rows.append((y, scalars + seq))
    import numpy as np
    y = np.array([r[0] for r in rows])
    X = np.array([r[1] for r in rows])
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_pkl", required=True)
    ap.add_argument("--preds", required=True)  # outputs/<run>.jsonl
    ap.add_argument("--K", type=int, default=32)
    args = ap.parse_args()

    X, y = build_features_from_jsonl(args.preds, K=args.K)
    clf = joblib.load(args.model_pkl)
    import numpy as np
    prob = clf.predict_proba(X)[:,1]

    src = Path(args.preds)
    out = src.with_suffix(".chair_scored.jsonl")
    with open(src, "r", encoding="utf-8") as f_in, open(out, "w", encoding="utf-8") as f_out:
        for p, line in zip(prob, f_in):
            r = json.loads(line)
            r["chair_score"] = float(p)  # predicted P(hallucination)
            f_out.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote: {out}\nMean chair_score={prob.mean():.3f}")

if __name__ == "__main__":
    main()
