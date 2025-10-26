import argparse, json
from pathlib import Path
import numpy as np
import sys

"""
SUMMARY:
Extracts features from eval_mc1.py's JSONL output for the NN classifier.
This version saves features to a .jsonl file, preserving the vector/sequence
structure needed for the attention model.
It also creates a .meta.json file to store scalar feature names.
"""

def fetch_args():
    ap = argparse.ArgumentParser()
    # This is the line that was fixed before
    ap.add_argument("--preds", required=True, help="outputs/*.jsonl from eval_mc1")
    ap.add_argument("--out", default="", help="Out .jsonl path (e.g., outputs/train_run.features.jsonl)")
    ap.add_argument("--K", type=int, default=32, help="Tail length for token logprob/entropy sequences")
    return ap.parse_args()
    
# Normalize and pad/truncate the last layer tokens
def pad_tail(xs, k, pad=0.0):
    """Pads or truncates a list to be of length k."""
    xs = list(xs)[-k:]
    if len(xs) < k: 
        xs = [pad]*(k-len(xs)) + xs
    return xs

# Extract features from a single record
def row_from_record(r, K=32):
    """
    Extracts a label (y) and a feature dictionary (feats) from a raw JSON record.
    Features are split into 'scalar' (single values) and 'sequence' (vectors).
    """
    # Label: 1 = hallucination (wrong), 0 = correct
    # We get this from tag_mc1.py, but fallback to 'correct' field if not present
    y = int(r.get("hallucination", not r.get("correct", False)))
    
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
    
    # Last layer token sequences (padded/truncated to K)
    last = r.get("last_layer", {})
    lp_seq  = pad_tail(last.get("logprobs_seq", []), K, 0.0)
    ent_seq = pad_tail(last.get("entropies_seq", []), K, 0.0)
    
    # This is the key change from featurize.py:
    # We store the sequences as actual lists (vectors)
    sequence_feats = {
        "last_lp_tail_vec": [float(v) for v in lp_seq],
        "last_ent_tail_vec": [float(v) for v in ent_seq]
    }
        
    return y, scalar_feats, sequence_feats

def main():
    args = fetch_args()

    # Set up paths
    src = Path(args.preds)
    if not args.out:
        print("Error: --out argument is required for featurize_nn.py")
        print("Example: --out outputs/train_run.features.jsonl")
        sys.exit(1)
        
    out_jsonl = Path(args.out)
    out_meta = out_jsonl.with_suffix(".meta.json")

    # Read input, extract features, write JSONL
    all_scalar_keys = set()
    total_rows = 0
    
    print(f"Starting featurization for K={args.K}...")
    with open(src, "r", encoding="utf-8") as f_in, \
         open(out_jsonl, "w", encoding="utf-8") as f_out:
        
        for line in f_in:
            line = line.strip()
            if not line: 
                continue
            
            r = json.loads(line)
            y, scalar_feats, sequence_feats = row_from_record(r, args.K)
            
            # Update the set of all scalar keys found
            all_scalar_keys.update(scalar_feats.keys())
            
            # Combine all features into one record for this line
            record_out = {
                "y": y,
                "scalar_features": scalar_feats,
                "sequence_features": sequence_feats
            }
            
            f_out.write(json.dumps(record_out, ensure_ascii=False) + "\n")
            total_rows += 1

    if total_rows == 0:
        print(f"Error: No rows parsed from input file: {src}")
        sys.exit(1)

    print(f"Successfully processed {total_rows} rows.")

    # Save the metadata (the ordered list of scalar feature names)
    # This is CRITICAL for the model to know the order of inputs
    sorted_scalar_keys = sorted(list(all_scalar_keys))
    
    # --- THIS IS THE FIX ---
    # The 'scalar_feature_names' key was missing in your old version
    meta = {
        "K": args.K,
        "n_scalar_features": len(sorted_scalar_keys),
        "scalar_feature_names": sorted_scalar_keys, 
        "sequence_feature_names": ["last_lp_tail_vec", "last_ent_tail_vec"]
    }
    # -----------------------
    
    with open(out_meta, "w", encoding="utf-8") as f_meta:
        json.dump(meta, f_meta, indent=2)

    # Note the new print statements, which you'll see in your log next time:
    print(f"Wrote features to: {out_jsonl}")
    print(f"Wrote metadata to: {out_meta}")

if __name__ == "__main__":
    main()

