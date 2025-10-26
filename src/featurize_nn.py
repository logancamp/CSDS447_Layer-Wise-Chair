import argparse, json, csv
from pathlib import Path
import re # Using re for sorting keys

"""
SUMMARY:
Post Processing: Extracts features from multiple-choice evaluation outputs
(from eval_mc1.py) and saves them in a .jsonl format suitable for
a neural network.

This is a modification of featurize.py. Instead of a flat CSV, it saves
features as a JSON object per line, preserving the vector structure of
token sequences.
"""

def fetch_args():
    ap = argparse.ArgumentParser()
    ap.add.argument("--preds", required=True, help="outputs/*.jsonl from eval_mc1")
    ap.add_argument("--out", default="", help="out JSONL path")
    ap.add_argument("--K", type=int, default=32, help="tail length for token logprob sequence")
    return ap.parse_args()
    
# Normalize and pad/truncate the last layer tokens
def pad_tail(xs, k, pad=0.0):
    xs = list(xs)[-k:]
    if len(xs) < k: xs = [pad]*(k-len(xs)) + xs
    return xs

# Helper to sort feature keys naturally (e.g., layer1, layer2, layer10)
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'([0-9]+)', s)]

# Extract features from a single record (default K=32 tokens)
def row_from_record(r, K=32):
    # Label: 1 = hallucination (wrong), 0 = correct
    y = int(r.get("hallucination", not r.get("correct", False)))
    
    # --- Scalar Features ---
    scalar_feats = {}
    hist = sorted(r.get("historical_layers", []), key=lambda d: d.get("layer", 0))
    for item in hist:
        L = item.get("layer")
        lp = item.get("lp_summary", {})
        en = item.get("ent_summary", {})
        # Separate features for logprobs and entropies
        for k, v in lp.items():
            scalar_feats[f"layer{L}_lp_{k}"] = float(v)
        for k, v in en.items():
            scalar_feats[f"layer{L}_ent_{k}"] = float(v)
    
    # --- Sequence Features (as vectors) ---
    last = r.get("last_layer", {})
    lp_seq  = pad_tail(last.get("logprobs_seq", []), K, 0.0)
    ent_seq = pad_tail(last.get("entropies_seq", []), K, 0.0)
    
    # Create the final feature dictionary
    features = {
        "scalar_features": scalar_feats,
        "last_lp_tail_vec": list(lp_seq),
        "last_ent_tail_vec": list(ent_seq),
    }
        
    return y, features

def main():
    args = fetch_args()

    # Set up paths
    src = Path(args.preds)
    out = Path(args.out) if args.out else src.with_suffix(".features.jsonl")

    # Read input, extract features, write JSONL
    rows = []
    all_scalar_keys = set()

    with open(src, "r", encoding="utf-8") as f_in:
        for line in f_in:
            line = line.strip()
            if not line: 
                continue
            r = json.loads(line)
            y, feats = row_from_record(r, args.K)
            rows.append({"y": y, "features": feats})
            all_scalar_keys.update(feats["scalar_features"].keys())

    if not rows:
        raise SystemExit("No rows parsed from input.")

    # We need a fixed, sorted order for the scalar features
    # so the model always sees them in the same order.
    sorted_scalar_keys = sorted(list(all_scalar_keys), key=natural_sort_key)
    
    # Save metadata (like the scalar key order)
    meta = {
        "K": args.K,
        "scalar_feature_keys": sorted_scalar_keys,
        "num_scalar_features": len(sorted_scalar_keys)
    }
    meta_path = out.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f_meta:
        json.dump(meta, f_meta, indent=2)
    print(f"Wrote feature metadata: {meta_path}")

    # Write the .jsonl file
    with open(out, "w", encoding="utf-8") as f_out:
        for row in rows:
            # Re-order scalar features dictionary based on the sorted keys
            scalar_dict = row["features"]["scalar_features"]
            sorted_scalars = [scalar_dict.get(k, 0.0) for k in sorted_scalar_keys]
            
            # Replace the dict with the ordered list
            row["features"]["scalar_features"] = sorted_scalars
            
            f_out.write(json.dumps(row) + "\n")

    print(f"Wrote features for NN training: {out}")

if __name__ == "__main__":
    main()
