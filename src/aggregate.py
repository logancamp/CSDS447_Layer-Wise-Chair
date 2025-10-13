# src/aggregate.py
import json, glob, statistics as stats
from pathlib import Path

def load_metrics(pattern="outputs/*.metrics.json"):
    rows=[]
    for p in glob.glob(pattern):
        with open(p,"r",encoding="utf-8") as f:
            m=json.load(f); m["_file"]=p; rows.append(m)
    return rows

def table(rows):
    # columns: model, n, acc, err, file
    lines=["model\tn\tacc\terr\tmetrics_path"]
    for r in sorted(rows, key=lambda x:(x["model"], -x["n"])):
        lines.append(f'{r["model"]}\t{r["n"]}\t{r["accuracy"]:.3f}\t{1-r["accuracy"]:.3f}\t{r["_file"]}')
    return "\n".join(lines)

def summarize(rows):
    by_model={}
    for r in rows: by_model.setdefault(r["model"], []).append(r["accuracy"])
    agg=[]
    for m, accs in by_model.items():
        agg.append({"model":m, "runs":len(accs), "mean_acc":stats.mean(accs), "stdev_acc":(stats.stdev(accs) if len(accs)>1 else 0.0)})
    return sorted(agg, key=lambda x: -x["mean_acc"])

if __name__=="__main__":
    rows=load_metrics()
    if not rows:
        print("No metrics found in outputs/. Run eval first.")
        raise SystemExit(0)
    print(table(rows))
    print("\n== Summary by model ==")
    for a in summarize(rows):
        print(f'{a["model"]}: runs={a["runs"]} mean_acc={a["mean_acc"]:.3f} stdev={a["stdev_acc"]:.3f}')
