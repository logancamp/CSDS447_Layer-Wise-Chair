import argparse, json, csv
from pathlib import Path

"""
SUMMARY:
Post Processing: Tags multiple-choice evaluation outputs with lexical overlap
and hallucination indicators, then writes enriched JSONL and CSV files.
Requires outputs from eval_mc1.py.

Used by train_chair for classifier training and analysis.
"""

def fetch_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="outputs/*.jsonl from eval_mc1")
    ap.add_argument("--out", default="", help="out CSV path")
    ap.add_argument("--K", type=int, default=16, help="tail length for token logprob sequence")
    return ap.parse_args()
    
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
    feats = {}
    hist = sorted(r.get("historical_layers", []), key=lambda d: d.get("layer", 0))
    for item in hist:
        L = item.get("layer")
        lp = item.get("lp_summary", {})
        en = item.get("ent_summary", {})
        # Separate features for logprobs and entropies
        for k, v in lp.items():
            feats[f"layer{L}_lp_{k}"] = float(v)
        for k, v in en.items():
            feats[f"layer{L}_ent_{k}"] = float(v)
    
    # Last layer token sequences (padded/truncated to K)
    last = r.get("last_layer", {})
    lp_seq  = pad_tail(last.get("logprobs_seq", []), K, 0.0)
    ent_seq = pad_tail(last.get("entropies_seq", []), K, 0.0)
    
    # For logistic model
    for i, v in enumerate(lp_seq, 1):
        feats[f"last_lp_tail_{i}"] = float(v)
    for i, v in enumerate(ent_seq, 1):
        feats[f"last_ent_tail_{i}"] = float(v)
    
    # Add overlap features if present (from tagged file)
    for k in ["overlap_pred", "overlap_correct", "overlap_max_distractor"]:
        if k in r:
            feats[k] = float(r[k])
            
    # For when we change to the attention model
    # feats["last_lp_tail_vec"]  = list(lp_seq)
    # feats["last_ent_tail_vec"] = list(ent_seq)
        
    return y, feats

def main():
    args = fetch_args()

    # Set up paths
    src = Path(args.preds)
    out = Path(args.out) if args.out else src.with_suffix(".features.csv")

    # Read input, extract features, write CSV
    rows = []
    all_keys = set()

    with open(src, "r", encoding="utf-8") as f_in:
        for line in f_in:
            line = line.strip()
            if not line: 
                continue
            r = json.loads(line)
            y, feats = row_from_record(r, args.K)
            rows.append((y, feats))
            all_keys |= feats.keys()

    if not rows:
        raise SystemExit("No rows parsed from input.")

    feat_keys = sorted(all_keys)
    with open(out, "w", newline="", encoding="utf-8") as f_out:
        w = csv.writer(f_out)
        w.writerow(["y"] + feat_keys)
        for y, feats in rows:
            w.writerow([y] + [feats.get(k, 0.0) for k in feat_keys])

    print(f"Wrote features for training: {out}")

if __name__ == "__main__":
    main()
