# src/tag_mc1.py
import argparse, json, re, csv
from pathlib import Path

def toks(s): return set(re.findall(r"\w+", s.lower()))

def overlap(a, b): return len(toks(a) & toks(b))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="path to eval_mc1 jsonl (outputs/*.jsonl)")
    args = ap.parse_args()

    in_path = Path(args.preds)
    out_jsonl = in_path.with_suffix(".tagged.jsonl")
    out_csv   = in_path.with_suffix(".tagged.csv")

    n=0; correct=0
    with in_path.open() as f_in, out_jsonl.open("w", encoding="utf-8") as f_out, out_csv.open("w", newline="", encoding="utf-8") as f_csv:
        w = csv.writer(f_csv)
        w.writerow(["question","pred_index","correct_index","pred_text","correct_text",
                    "is_correct","hallucination","ovlp_pred","ovlp_correct","ovlp_max_distractor"])
        for line in f_in:
            ex = json.loads(line)
            q        = ex["question"]
            choices  = ex["choices"]
            labels   = ex["labels"]
            gen      = ex["generated"]
            pred_i   = ex["pred_index"]
            corr_i   = labels.index(1)

            ovlp_correct = overlap(gen, choices[corr_i])
            ovlp_pred    = overlap(gen, choices[pred_i])
            ovlp_distr   = max(overlap(gen, c) for j,c in enumerate(choices) if j != corr_i)

            is_corr = (pred_i == corr_i)
            hallu   = not is_corr  # CHAIR-lite proxy: wrong ⇒ hallucination

            ex_out = {**ex,
                "correct_index": corr_i,
                "correct_text": choices[corr_i],
                "overlap_pred": ovlp_pred,
                "overlap_correct": ovlp_correct,
                "overlap_max_distractor": ovlp_distr,
                "hallucination": hallu,
            }
            f_out.write(json.dumps(ex_out, ensure_ascii=False) + "\n")
            w.writerow([q, pred_i, corr_i, choices[pred_i], choices[corr_i],
                        is_corr, hallu, ovlp_pred, ovlp_correct, ovlp_distr])

            n += 1; correct += int(is_corr)

    print(f"Tagged {n} examples | acc={correct/n:.3f} | wrote:\n- {out_jsonl}\n- {out_csv}")

if __name__ == "__main__":
    main()
