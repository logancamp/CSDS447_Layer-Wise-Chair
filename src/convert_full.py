import argparse
import os
import pandas as pd


def normalize(series: pd.Series) -> pd.Series:
    mn = series.min()
    mx = series.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        # avoid divide-by-zero; return zeros if constant
        return series * 0.0
    return (series - mn) / (mx - mn)


def create_chair_from_features(df: pd.DataFrame, out_dir: str, tails=None):
    os.makedirs(out_dir, exist_ok=True)

    # label column in your file is "y"
    if "y" not in df.columns:
        raise ValueError(f"Expected label column 'y' not found. Columns: {list(df.columns)[:30]}")
    label_col = "y"

    # by default, just use tail_1 (last token)
    if tails is None:
        tails = [1]

    # which feature columns to carry through *in addition* to lp/ent + norms
    # (assumes your extra features are already computed in df)
    meta_cols = {"qid", "split", label_col}
    # everything that isn't pure metadata is treated as a feature we can pass through
    passthrough_feature_cols = [c for c in df.columns if c not in meta_cols]

    for t in tails:
        # default: use last-layer summary columns
        lp_col = f"last_lp_tail_{t}"
        ent_col = f"last_ent_tail_{t}"

        if lp_col not in df.columns or ent_col not in df.columns:
            print(f"[skip] missing {lp_col} or {ent_col}")
            continue

        # base CHAIR-style features
        out = pd.DataFrame(index=df.index)
        out["lp"] = df[lp_col]
        out["ent"] = df[ent_col]
        out["lp_norm"] = normalize(out["lp"])
        out["ent_norm"] = normalize(out["ent"])

        # add all existing feature columns (including other tails / layer stats / histograms, etc.)
        # this gives you a superset: our features + CHAIR-style ones
        # build passthrough cols in one DataFrame to avoid fragmentation
        pt_df = df[passthrough_feature_cols].copy()

        # avoid overwriting lp/ent/lp_norm/ent_norm
        pt_df = pt_df[[c for c in pt_df.columns if c not in out.columns]]

        # concatenate once
        out = pd.concat([out, pt_df], axis=1)

        # keep id/split/label
        if "qid" in df.columns:
            out["qid"] = df["qid"]
        if "split" in df.columns:
            out["split"] = df["split"]
        else:
            out["split"] = "train"

        out["y"] = df[label_col]
        fname = f"chair_features.csv"

        out_path = os.path.join(out_dir, fname)
        out.to_csv(out_path, index=False)
        print(f"[ok] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--features",
        required=True,
        help="Path to mc1_results.features.csv",
    )
    ap.add_argument(
        "--base_features",
        help=(
            "Optional path to a base features CSV that contains qid, y, and split. "
            "If provided and the --features file does not have y/split, we will merge them by qid."
        ),
    )
    ap.add_argument(
        "--out",
        default="chair_data",
        help="Output directory for CHAIR-style CSVs",
    )
    args = ap.parse_args()

    # primary features file (may be either last-layer features or historical-layer stats)
    df = pd.read_csv(args.features)

    # If the main file does not already contain labels/splits, optionally merge from base_features
    if ("y" not in df.columns or "split" not in df.columns) and args.base_features is not None:
        base_df = pd.read_csv(args.base_features)
        required_cols = ["qid", "y", "split"]
        missing = [c for c in required_cols if c not in base_df.columns]
        if missing:
            raise ValueError(f"base_features is missing required columns: {missing}")

        df = df.merge(base_df[required_cols], on="qid", how="left")
        if df["y"].isna().any():
            raise ValueError("Some rows have no label after merging base_features. Check qid alignment.")

    # choose a single "full-tail" summary:
    # for last-layer features, pick the largest available tail index from last_lp_tail_*
    # for a specific historical layer, pick the largest available tail index from lp_L{layer}_tail_*
    prefix = "last_lp_tail_"
    tail_ids = sorted(
        int(c.split("_")[-1])
        for c in df.columns
        if c.startswith(prefix)
    )
    if not tail_ids:
        raise ValueError(
            f"No last-layer tail columns like {prefix}* found in {args.features}"
        )
    tails = [tail_ids[-1]]

    create_chair_from_features(df, args.out, tails=tails)


if __name__ == "__main__":
    main()