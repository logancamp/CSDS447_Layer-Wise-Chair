# src/baselines.py
import argparse, random
from datasets import load_dataset

def acc_random(ds, label_path=("mc1_targets","labels")) -> float:
    rng = random.Random(0)
    correct = 0
    for ex in ds:
        labels = ex[label_path[0]][label_path[1]]
        guess = rng.randrange(len(labels))
        correct += 1 if labels[guess] == 1 else 0
    return correct / len(ds)

def acc_first(ds, label_path=("mc1_targets","labels")) -> float:
    # Simple proxy “majority”: always pick index 0 (MC1 has 1 true among choices)
    correct = 0
    for ex in ds:
        labels = ex[label_path[0]][label_path[1]]
        correct += 1 if labels[0] == 1 else 0
    return correct / len(ds)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", choices=["random","first"], required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = use full split")
    ap.add_argument("--split", default="validation")
    args = ap.parse_args()

    ds = load_dataset("truthful_qa","multiple_choice")[args.split]
    if args.limit and args.limit < len(ds):
        ds = ds.select(range(args.limit))

    if args.baseline == "random":
        acc = acc_random(ds)
    else:
        acc = acc_first(ds)

    print(f"{args.baseline} accuracy (MC1, {len(ds)} ex): {acc:.3f}")

if __name__ == "__main__":
    main()
