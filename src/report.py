# src/report.py
import argparse, json, statistics as stats
from pathlib import Path

def load_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def ensure_tagged(base_path: Path) -> Path:
    tp = base_path.with_suffix(".tagged.jsonl")
    if tp.exists():
        return tp
    # Fallback: derive minimal tags from base eval jsonl (no overlap columns)
    tagged = []
    for ex in load_jsonl(base_path):
        corr_i = ex["labels"].index(1)
        ex["correct_index"] = corr_i
        ex["correct_text"]  = ex["choices"][corr_i]
        ex["hallucination"] = (ex["pred_index"] != corr_i)
        tagged.append(ex)
    with open(tp, "w", encoding="utf-8") as f:
        for e in tagged:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return tp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="outputs/*.jsonl or *.tagged.jsonl")
    ap.add_argument("--topk", type=int, default=10, help="show worst K examples")
    args = ap.parse_args()

    base = Path(args.preds)
    tagged_path = base if base.suffixes[-2:] == [".tagged",".jsonl"] else ensure_tagged(base)

    rows = list(load_jsonl(tagged_path))
    n = len(rows)
    acc = sum(int(r.get("correct", r["pred_index"]==r["correct_index"])) for r in rows) / max(n,1)
    hallu = sum(int(r.get("hallucination", r["pred_index"]!=r["correct_index"])) for r in rows) / max(n,1)

    # If overlap fields exist, compute simple deltas
    deltas = []
    for r in rows:
        if "overlap_pred" in r and "overlap_correct" in r:
            deltas.append(r["overlap_correct"] - r["overlap_pred"])
    delta_stats = {
        "count": len(deltas),
        "mean": stats.mean(deltas) if deltas else 0.0,
        "stdev": stats.stdev(deltas) if len(deltas)>1 else 0.0,
    }

    # Pick worst mistakes (wrong + highest overlap with distractors if available)
    def mistake_key(r):
        if r.get("hallucination", r["pred_index"]!=r["correct_index"]):
            if "overlap_max_distractor" in r:
                return r["overlap_max_distractor"]
            return 0
        return -1
    worst = sorted(
        (r for r in rows if r.get("hallucination", r["pred_index"]!=r["correct_index"])),
        key=mistake_key, reverse=True
    )[:args.topk]

    # Write a markdown report next to inputs
    out_md = tagged_path.with_suffix(".report.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(f"# Report for `{tagged_path.name}`\n\n")
        f.write(f"- N: **{n}**\n- Accuracy: **{acc:.3f}**\n- Hallucination-rate (wrong): **{hallu:.3f}**\n")
        if deltas:
            f.write(f"- Overlap Δ (correct - pred): mean={delta_stats['mean']:.2f} ± {delta_stats['stdev']:.2f} (n={delta_stats['count']})\n")
        f.write("\n## Worst mistakes\n")
        for i, r in enumerate(worst, 1):
            q = r["question"].replace("\n"," ")[:300]
            pred_i, corr_i = r["pred_index"], r["correct_index"]
            pred = r["choices"][pred_i]; corr = r["choices"][corr_i]
            f.write(f"\n**{i}. Q:** {q}\n\n")
            f.write(f"- **Pred:** {pred}\n- **Gold:** {corr}\n")
            if "overlap_pred" in r:
                f.write(f"- overlaps: pred={r['overlap_pred']} | correct={r['overlap_correct']} | max_distr={r.get('overlap_max_distractor','-')}\n")
            f.write(f"- generated: {r['generated'][:400].replace(chr(10),' ')}\n")

    # Also emit a tiny metrics JSON
    out_json = tagged_path.with_suffix(".report.metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "n": n,
            "accuracy": acc,
            "hallucination_rate": hallu,
            "overlap_delta": delta_stats,
            "source": str(tagged_path)
        }, f, indent=2)
    print(f"Wrote:\n- {out_md}\n- {out_json}")

if __name__ == "__main__":
    main()
