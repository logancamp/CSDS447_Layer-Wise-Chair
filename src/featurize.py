import argparse, json, csv
from pathlib import Path

"""
SUMMARY:
Post Processing: Tags multiple-choice evaluation outputs with lexical overlap
and hallucination indicators, then writes enriched JSONL and CSV files.
Requires outputs from eval_mc1.py.

Used by train_chair for classifier training and analysis.
"""

# Normalize and pad/truncate the last layer tokens
def pad_tail(xs, k, pad=0.0):
    xs = list(xs)[-k:]
    if len(xs) < k: xs = [pad]*(k-len(xs)) + xs
    return xs

# Extract features from a single record (default K=32 tokens)
def row_from_record(r, K=32):
    # Label: 1 = hallucination (wrong), 0 = correct
    y = int(r.get("hallucination", not r.get("correct", False)))
    
    # Scalar summaries from eval_mc1
    scalars = {
        "avg_logprob": r.get("avg_logprob", 0.0),
        "avg_prob": r.get("avg_prob", 0.0),
        "perplexity": r.get("perplexity", 0.0),
        "lp_mean": r.get("lp_mean", 0.0),
        "lp_std": r.get("lp_std", 0.0),
        "lp_min": r.get("lp_min", 0.0),
        "lp_max": r.get("lp_max", 0.0),
        "lp_first": r.get("lp_first", 0.0),
        "lp_last": r.get("lp_last", 0.0),
        "lp_delta": r.get("lp_delta", 0.0),
        "ent_mean": r.get("ent_mean", 0.0),
        "ent_std": r.get("ent_std", 0.0),
        "ent_min": r.get("ent_min", 0.0),
        "ent_max": r.get("ent_max", 0.0),
        "ent_first": r.get("ent_first", 0.0),
        "ent_last": r.get("ent_last", 0.0),
        "ent_delta": r.get("ent_delta", 0.0),
    }
    seq = r.get("token_logprobs", [])
    seq = pad_tail(seq, K, 0.0)
    return y, scalars, seq

def main():
    # Add command-line args
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="outputs/*.jsonl from eval_mc1")
    ap.add_argument("--out", default="", help="out CSV path")
    ap.add_argument("--K", type=int, default=32, help="tail length for token logprob sequence")
    args = ap.parse_args()

    # Set up paths
    src = Path(args.preds)
    out = Path(args.out) if args.out else src.with_suffix(".features.csv")

    # Read input, extract features, write CSV
    with open(src, "r", encoding="utf-8") as f_in, open(out, "w", newline="", encoding="utf-8") as f_out:
        rows = []
        for line in f_in:
            if not line.strip(): continue
            r = json.loads(line)
            y, scalars, seq = row_from_record(r, args.K)
            row = {"y": y, **scalars}
            row.update({f"lp_t{-i}": v for i, v in enumerate(range(args.K,0,-1), start=1)})  # headers only
            rows.append((y, scalars, seq))

        # Build header properly
        header = ["y"] + list(rows[0][1].keys()) + [f"lp_tail_{i}" for i in range(1, args.K+1)]
        w = csv.writer(f_out)
        w.writerow(header)
        for y, scalars, seq in rows:
            w.writerow([y] + list(scalars.values()) + seq)

    print(f"Wrote features: {out}")

if __name__ == "__main__":
    main()
