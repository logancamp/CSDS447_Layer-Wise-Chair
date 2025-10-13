# src/aggregate.py (drop-in)
import json, glob, statistics as stats
from pathlib import Path

REQ = ["n", "accuracy"]  # minimal required keys

def load_metrics(pattern="outputs/*.metrics.json"):
    rows=[]
    for p in glob.glob(pattern):
        try:
            with open(p,"r",encoding="utf-8") as f:
                m=json.load(f); m["_file"]=p
                if all(k in m for k in REQ):
                    rows.append(m)
        except Exception:
            # skip unreadable files
            pass
    return rows

def table(rows):
    lines=["model\tn\tacc\terr\tmetrics_path"]
    # safe fields with defaults
    def keyfn(r):
        model = r.get("model","<unknown>")
        n = r.get("n", 0)
        return (model, -int(n))
    for r in sorted(rows, key=keyfn):
        model = r.get("model","<unknown>")
        n = int(r.get("n",0))
        acc = float(r.get("accuracy",0.0))
        lines.append(f"{model}\t{n}\t{acc:.3f}\t{1-acc:.3f}\t{r['_file']}")
    return "\n".join(lines)

def summarize(rows):
    by_model={}
    for r in rows:
        model = r.get("model","<unknown>")
        by_model.setdefault(model, []).append(float(r.get("accuracy",0.0)))
    agg=[]
    for m, accs in by_model.items():
        if not accs: continue
        mu = stats.mean(accs)
        sd = stats.stdev(accs) if len(accs)>1 else 0.0
        agg.append({"model":m,"runs":len(accs),"mean_acc":mu,"stdev_acc":sd})
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
