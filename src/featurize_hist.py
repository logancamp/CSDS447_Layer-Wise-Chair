import argparse, json, csv
from pathlib import Path

def pad_tail(xs, k, pad=0.0):
    xs = list(xs)[-k:]
    if len(xs) < k: xs = [pad]*(k-len(xs)) + xs
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

            # For each historical layer, emit tailed sequences
            for li, L in enumerate(idxs):
                lp_seq = pad_tail([float(v) for v in lp_layers[li]], args.K, 0.0)
                ent_seq = pad_tail([float(v) for v in ent_layers[li]], args.K, 0.0)

                for i, v in enumerate(lp_seq, 1):
                    feats[f"hist_lp_L{L}_tail_{i}"] = v
                for i, v in enumerate(ent_seq, 1):
                    feats[f"hist_ent_L{L}_tail_{i}"] = v

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