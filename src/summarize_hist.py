import argparse
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from logit_utils import summarize_entropies, summarize_logprobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_csv",
        required=True,
        help="Path to mc1_results.historical_layers.csv",
    )
    ap.add_argument(
        "--out_csv",
        required=True,
        help="Path to write summarized historical features CSV",
    )
    ap.add_argument(
        "--omit_last_k_layers",
        type=int,
        default=0,
        help=(
            "Number of highest layers to drop when summarizing across layers. "
            "0 = use all layers present."
        ),
    )
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    # Identify per-layer columns: L{layer}_{ent|lp}_tail_{t}
    pat = re.compile(r"^L(\d+)_(ent|lp)_tail_(\d+)$")
    layer_cols = []  # list of (col_name, layer, metric, tail)
    for c in df.columns:
        m = pat.match(c)
        if m:
            layer = int(m.group(1))
            metric = m.group(2)  # 'ent' or 'lp'
            tail = int(m.group(3))
            layer_cols.append((c, layer, metric, tail))

    if not layer_cols:
        raise ValueError("No L{layer}_{ent|lp}_tail_{t} columns found in input CSV.")

    all_layers = sorted({L for _, L, _, _ in layer_cols})
    max_layer = max(all_layers)
    k = max(0, args.omit_last_k_layers)
    cutoff = max_layer - k

    # Ensure we keep at least one layer
    include_layers = [L for L in all_layers if L <= cutoff]
    if not include_layers:
        include_layers = [max_layer]

    print(f"Total layers: {len(all_layers)}, using layers <= {cutoff} → {include_layers}")

    # Group columns by (metric, tail)
    cols_by_key = defaultdict(list)  # (metric, tail) -> list of (layer, col_name)
    for cname, L, metric, tail in layer_cols:
        if L in include_layers:
            cols_by_key[(metric, tail)].append((L, cname))

    # Sort by layer within each group
    for key in cols_by_key:
        cols_by_key[key].sort(key=lambda x: x[0])

    # Start output with metadata
    meta_cols = [c for c in ["qid", "split", "y"] if c in df.columns]
    out = df[meta_cols].copy()

    # Effective "last layer" after applying K: highest included layer
    effective_last_layer = max(include_layers)

    # Collect all tails we have summaries for
    tails_present = sorted({tail for (_, tail) in cols_by_key.keys()})

    # ----- Build / preserve last_* features -----
    # k here is the *number of last layers omitted* (from args.omit_last_k_layers)
    if args.omit_last_k_layers == 0:
        # For k = 0, we want to reproduce the original features exactly.
        # If the input already has last_* columns, just pass them through.
        existing_last_cols = [c for c in df.columns if c.startswith("last_")]
        if existing_last_cols:
            for c in existing_last_cols:
                out[c] = df[c]
        else:
            # Fallback: construct last_* from the true last layer if none exist.
            for tail in tails_present:
                ent_col = f"L{effective_last_layer}_ent_tail_{tail}"
                lp_col = f"L{effective_last_layer}_lp_tail_{tail}"

                if ent_col not in df.columns or lp_col not in df.columns:
                    raise ValueError(
                        f"Missing {ent_col} or {lp_col} in input CSV; cannot build last-layer features."
                    )

                out[f"last_ent_tail_{tail}"] = df[ent_col]
                out[f"last_lp_tail_{tail}"] = df[lp_col]
    else:
        # For k > 0, intentionally redefine "last_*" to mean the highest
        # *included* layer after omitting the top k layers.
        for tail in tails_present:
            ent_col = f"L{effective_last_layer}_ent_tail_{tail}"
            lp_col = f"L{effective_last_layer}_lp_tail_{tail}"

            if ent_col not in df.columns or lp_col not in df.columns:
                raise ValueError(
                    f"Missing {ent_col} or {lp_col} in input CSV; cannot build offset last-layer features."
                )

            out[f"last_ent_tail_{tail}"] = df[ent_col]
            out[f"last_lp_tail_{tail}"] = df[lp_col]

    # Build cross-layer summaries using the SAME util funcs as eval (summarize_*).
    for (metric, tail), LC in cols_by_key.items():
        col_names = [c for (_, c) in LC]
        vals = df[col_names].to_numpy(dtype=float)  # shape: (n_examples, n_layers_used)

        summaries = []
        for row in vals:
            # row is the per-layer values for this (metric, tail) and example
            trace = [v for v in row if not np.isnan(v)]
            if not trace:
                # fall back to zeros if everything is NaN
                if metric == "ent":
                    d = summarize_entropies([0.0])
                else:
                    d = summarize_logprobs([0.0])
            else:
                if metric == "ent":
                    d = summarize_entropies(trace)
                else:
                    d = summarize_logprobs(trace)
            summaries.append(d)

        # Use keys from the first summary dict to define columns
        # Use keys from the first summary dict to define columns
        keys = list(summaries[0].keys())

        # Build all summary columns at once to avoid fragmentation warnings
        summary_cols = {}
        for key in keys:
            col_name = f"{metric}_{key}_tail_{tail}"
            summary_cols[col_name] = [d[key] for d in summaries]

        summary_df = pd.DataFrame(summary_cols)
        out = pd.concat([out, summary_df], axis=1)

    out.to_csv(args.out_csv, index=False)
    print(f"[ok] wrote {args.out_csv}")


if __name__ == "__main__":
    main()