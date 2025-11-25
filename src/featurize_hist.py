import argparse, json, csv
from pathlib import Path

def pad_tail(xs, k, pad=0.0):
    xs = list(xs)[-k:]
    if len(xs) < k:
        xs = [pad] * (k - len(xs)) + xs
    return xs

def fetch_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="Path to eval_mc1 JSONL")
    ap.add_argument("--K", type=int, default=16, help="Tail length per layer")
    return ap.parse_args()

def main():
    args = fetch_args()
    src = Path(args.preds)
    out = src.with_suffix(".historical_layers.csv")

    rows = []
    all_keys = set()

    with open(src, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)

            # Label and meta, matching featurize.py
            y = int(r.get("hallucination", False))
            qid = r.get("qid", "")
            split = r.get("split", "")

            hist = r.get("historical_layers", None)
            if not hist:
                # Skip records without historical layers (compat with old runs)
                continue

            idxs = hist.get("layer_indices_in_range", [])
            lp_layers = hist.get("logprobs_seq_by_layer", [])
            ent_layers = hist.get("entropies_seq_by_layer", [])

            # Basic shape/length guard
            if not (len(idxs) == len(lp_layers) == len(ent_layers)):
                continue

            feats = {}

            # ----------------------------
            # Per-layer historical tails
            # ----------------------------
            # For each historical layer, emit tailed sequences in the same
            # lp_tail_i / ent_tail_i style as featurize's last layer.
            for li, L in enumerate(idxs):
                lp_seq = pad_tail([float(v) for v in lp_layers[li]], args.K, 0.0)
                ent_seq = pad_tail([float(v) for v in ent_layers[li]], args.K, 0.0)

                for i, v in enumerate(lp_seq, 1):
                    # Example: L10_lp_tail_1, L10_lp_tail_2, ...
                    feats[f"L{L}_lp_tail_{i}"] = float(v)
                for i, v in enumerate(ent_seq, 1):
                    # Example: L10_ent_tail_1, L10_ent_tail_2, ...
                    feats[f"L{L}_ent_tail_{i}"] = float(v)

            last = r.get("last_layer", {})
            lp_seq_ll = last.get("logprobs_seq") or []
            ent_seq_ll = last.get("entropies_seq") or []

            if lp_seq_ll and ent_seq_ll:
                lp_seq_last = pad_tail([float(v) for v in lp_seq_ll], args.K, 0.0)
                ent_seq_last = pad_tail([float(v) for v in ent_seq_ll], args.K, 0.0)
            else:
                # If last_layer is missing, just fill zeros to keep columns aligned
                lp_seq_last = [0.0] * args.K
                ent_seq_last = [0.0] * args.K

            for i, v in enumerate(lp_seq_last, 1):
                feats[f"last_lp_tail_{i}"] = float(v)
            for i, v in enumerate(ent_seq_last, 1):
                feats[f"last_ent_tail_{i}"] = float(v)

            rows.append((qid, split, y, feats))
            all_keys |= set(feats.keys())

    if not rows:
        raise SystemExit("No rows with historical layers found.")

    feat_keys = sorted(all_keys)
    with open(out, "w", newline="", encoding="utf-8") as fout:
        w = csv.writer(fout)
        w.writerow(["qid", "split", "y"] + feat_keys)
        for qid, split, y, feats in rows:
            w.writerow([qid, split, y] + [feats.get(k, 0.0) for k in feat_keys])

    print(f"Wrote historical layer features: {out}")

if __name__ == "__main__":
    main()