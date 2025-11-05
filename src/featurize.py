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
    ap.add_argument("--K", type=int, default=16, help="tail length for token logprob sequence")
    return ap.parse_args()
    
# Normalize and pad/truncate the last layer tokens
def pad_tail(xs, k, pad=0.0):
    xs = list(xs)[-k:]
    if len(xs) < k: xs = [pad]*(k-len(xs)) + xs
    return xs

# Extract features from a single record (default K=16 tokens)
def row_from_record(r, K=16):
    # Label: 1 = hallucination (wrong), 0 = correct
    y = int(r["hallucination"])
    
    qid = r.get("qid", "")
    if not qid:
        raise ValueError("Missing 'qid' in record; ensure eval writes it.")

    split = r.get("split")
    if split is None:
        raise ValueError("Missing 'split' in record; ensure eval writes it.")
    
    # Scalar summaries from eval_mc1
    feats = {}
    traces = sorted(r.get("chair_token_traces", []), key=lambda d: d.get("t", 0))
    stats = ("mean", "std", "min", "max", "slope")
    for item in stats:
        lp_seq = [float((tok.get("lp_summary")  or {}).get(item, 0.0)) for tok in traces]
        ent_seq = [float((tok.get("ent_summary") or {}).get(item, 0.0)) for tok in traces]

        # fixed-length tails
        lp_seq = pad_tail(lp_seq, K, 0.0)
        ent_seq = pad_tail(ent_seq, K, 0.0)

        for i, v in enumerate(lp_seq, 1):
            feats[f"lp_{item}_tail_{i}"] = v
        for i, v in enumerate(ent_seq, 1):
            feats[f"ent_{item}_tail_{i}"] = v
    
    # Last layer token sequences (padded/truncated to K)
    last = r.get("last_layer", {})
    lp_seq_ll  = last.get("logprobs_seq", [])
    ent_seq_ll = last.get("entropies_seq", [])

    if lp_seq_ll and ent_seq_ll:
        lp_seq = pad_tail([float(v) for v in lp_seq_ll],  K, 0.0)
        ent_seq = pad_tail([float(v) for v in ent_seq_ll], K, 0.0)
    else:
        # Fallback: use the 'mean' trace tails you already emitted as a proxy
        lp_seq = [feats.get(f"lp_mean_tail_{i}", 0.0)  for i in range(1, K+1)]
        ent_seq = [feats.get(f"ent_mean_tail_{i}", 0.0) for i in range(1, K+1)]

    # Emit per-step columns so the trainer detects sequences (same for nn/lr)
    for i, v in enumerate(lp_seq, 1):
        feats[f"last_lp_tail_{i}"] = float(v)
    for i, v in enumerate(ent_seq, 1):
        feats[f"last_ent_tail_{i}"] = float(v)
    # ---------------------------------------------------------------
            
    return y, feats, {"qid": qid, "split": split}

def main():
    args = fetch_args()

    # Set up paths
    src = Path(args.preds)
    out = src.with_suffix(".features.csv")

    # Read input, extract features, write CSV
    rows = []
    all_keys = set()
    with open(src, "r", encoding="utf-8") as f_in:
        for line in f_in:
            line = line.strip()
            if not line: 
                continue
            r = json.loads(line)
            y, feats, meta = row_from_record(r, args.K)
            rows.append((y, feats, meta))
            all_keys |= feats.keys()

    if not rows:
        raise SystemExit("No rows parsed from input.")

    feat_keys = sorted(all_keys)
    with open(out, "w", newline="", encoding="utf-8") as f_out:
        w = csv.writer(f_out)
        w.writerow(["qid", "split", "y"] + feat_keys)
        for y, feats, meta in rows:
             w.writerow([meta["qid"], meta["split"], y] + [feats.get(k, 0.0) for k in feat_keys])

    print(f"Wrote features: {out}")

if __name__ == "__main__":
    main()
