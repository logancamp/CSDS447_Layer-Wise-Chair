import argparse, json, time
from pathlib import Path
from datasets import load_dataset # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from logit_utils import summarize_logprobs, summarize_entropies, hidden_to_token_logprobs, hidden_to_entropies
from typing import cast
from transformers.generation.utils import GenerateDecoderOnlyOutput
import hashlib
from collections import defaultdict

import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# --- speed knobs ---
import os
os.environ["TRANSFORMERS_CACHE"] = "/home/logancamp/.cache/huggingface"
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


"""
SUMMARY:
Evaluate a language model on TruthfulQA multiple-choice (mc1) questions.
Runs the model, gets the raw text output, picks a multiple-choice answer (via log-likelihood), and measures accuracy.
Stores raw results in a JSONL file, one entry per question, with various confidence metrics.
Later this data is tagged by tag_mc1 for training and pre-processed in predict_chair.
"""

# System prompt to use for all questions
# SYS = "You are concise and truthful. Answer the question factually and directly."
SYS = ""

def fetch_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()

def split_bucket(qid: str, p_train=0.72, p_val=0.18, p_test=0.10) -> str:
    h = int(hashlib.sha1(qid.encode()).hexdigest(), 16) % 10_000
    r = h / 10_000.0
    if r < p_train: return "train"
    if r < p_train + p_val: return "val"
    return "test"

# Return the index of the candidate with the highest log-likelihood: Same as TruthfulQA
def pick_by_loglikelihood(mdl, tok, question, choices, dev, policy="sum", alpha=0.7, prior_prompt=None):
    base = f"{SYS}\nQ: {question}\nA: "
    base_ids = tok(base, return_tensors="pt").to(dev)
    P = base_ids["input_ids"].shape[1]

    def score_ll(txt, mask_len):
        ids = tok(txt, return_tensors="pt").to(dev)
        labels = ids["input_ids"].clone()
        labels[:, :mask_len] = -100
        with torch.inference_mode():
            loss = mdl(**ids, labels=labels).loss.item()
        T = int((labels != -100).sum().item())
        total = -loss * max(T, 1)
        if policy == "sum":   return total
        if policy == "mean":  return total / max(T, 1)
        if policy == "alpha": return total / (max(T, 1) ** alpha)
        return total  # default

    # Choose PMI scorer implementation
    if policy == "pmi":
        prior = prior_prompt if prior_prompt is not None else "A:"
        prior_ids = tok(prior, return_tensors="pt").to(dev)
        PP = prior_ids["input_ids"].shape[1]
        pmi_ll_fn = lambda choice: score_ll(prior + " " + choice.strip(), PP)
    else:
        pmi_ll_fn = lambda _: 0.0

    mdl.eval()
    with torch.inference_mode():
        scores = []
        for c in choices:
            choice_txt = c.strip()
            s_main = score_ll(base + " " + choice_txt, P)
            s = s_main - pmi_ll_fn(choice_txt) if policy == "pmi" else s_main
            scores.append(s)

    return int(torch.as_tensor(scores).argmax().item())

def diagnostic_check():
    preds = [json.loads(l) for l in open("outputs/mc1_results.jsonl")]
    df = pd.DataFrame(preds)

    # --- Order invariance ---
    # If you re-ran with shuffled choices, compare predictions
    # print accuracy difference or disagreement rate between runs

    # --- Length bias check ---
    df["choice_lens"] = df["choices"].apply(lambda xs: [len(c.split()) for c in xs])
    df["pred_len"] = df.apply(lambda r: r["choice_lens"][r["pred_index"]], axis=1)
    r, _ = pearsonr(df["pred_len"], df["correct"].astype(int))
    print(f"Length-correlation (len vs correct): {r:.3f}")

    # --- Score-length correlation (if you saved scores) ---
    # For the new scorer returning diagnostics
    if "diagnostic" in df.columns and df["diagnostic"].notna().any():
        df_scores = pd.DataFrame(df["diagnostic"].dropna().tolist())
        if {"lengths","scores"}.issubset(df_scores.columns) and len(df_scores) > 1:
            print("Length–score correlation:", np.corrcoef(df_scores["lengths"], df_scores["scores"])[0,1])

    # --- Basic accuracy ---
    acc = df["correct"].mean()
    print(f"Accuracy: {acc:.3f}")

def main():
    # Configure hyperparameters
    args = fetch_args()
    
    model = "meta-llama/Meta-Llama-3-8B-Instruct"
    # model = "meta-llama/Llama-3.2-1B-Instruct"
    
    # model = "Qwen/Qwen3-4B-Instruct-2507"
    # model = "Qwen/Qwen3-4B-Thinking-2507"
    # model = "Qwen/Qwen3-8B"
    # model = "mistralai/Ministral-8B-Instruct-2410"
    
    max_new_tokens = 32

    # Set up output paths
    outdir = Path("outputs")
    outdir.mkdir(parents=True, exist_ok=True)
    outname = "mc1_results"
    pred_path = outdir / f"{outname}.jsonl"
    met_path = outdir / f"{outname}.metrics.json"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    # Check if cuda is available for device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Load model and tokenizer
    tok = AutoTokenizer.from_pretrained(model, use_fast=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"

    # Load the LLM model (8B on RTX 3070 via 4-bit + auto offload)
    mdl = AutoModelForCausalLM.from_pretrained(
        model,
        load_in_4bit=True,                               # quantize to 4-bit
        bnb_4bit_compute_dtype=torch.float16,            # fp16 compute
        device_map="auto",                               # GPU + CPU mapping
        offload_folder="/home/logancamp/offload",        # SSD folder in WSL
        low_cpu_mem_usage=True,
    )
    mdl.config.pad_token_id = tok.pad_token_id
    mdl.config.use_cache = True
    mdl.eval()

    # Optional: print placement to verify
    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    try:
        print("Device map:", mdl.hf_device_map)
    except Exception:
        pass


    # Load full split once
    ds = load_dataset("truthful_qa", "multiple_choice")["validation"]
    
    """ Each example in the dataset looks like this:
    {
    "question": "...",
    "mc1_targets": {
            "choices": ["A...", "B...", "C..."],
            "labels": [0, 1, 0]  # one correct answer
        }
    }
    """

    total = len(ds)
    correct = 0
    by_split = defaultdict(lambda: {"n": 0, "correct": 0})
    with open(pred_path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            
            # Prepare the prompt
            q = ex["question"]
            prompt = f"{SYS}\nQ: {q}\nA: "
            inputs = tok(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}  
            
            # Generate output
            print(f"Processing example {i+1}/{total}...", end="\r")
            with torch.inference_mode():
                out = mdl.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=1,
                    pad_token_id=tok.pad_token_id,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    use_cache=True,
                    do_sample=False, 
                    temperature=None, 
                    top_p=None, 
                    top_k=None
                )

            # Process output
            out = cast(GenerateDecoderOnlyOutput, out)
            prompt_len = inputs["input_ids"].shape[1]
            gen_ids = out.sequences[:, prompt_len:]
            gen_text = tok.batch_decode(gen_ids, skip_special_tokens=True)

            steps  = out.hidden_states
            assert steps is not None, "Hidden states missing. Did you set output_hidden_states=True?"
            
            num_layers = len(steps[0])
            FINAL_L = num_layers - 1

            # Final layer sequences
            last_ent_seq = hidden_to_entropies(steps, mdl, layer=FINAL_L)
            last_lp_seq = hidden_to_token_logprobs(steps, mdl, gen_ids, layer=FINAL_L)

            # All historical layer entropies and logprobs
            num_layers = len(steps[0])
            hist_start, hist_end = 1, FINAL_L
            
            ent_layers = []
            lp_layers = []
            for li in range(hist_start, hist_end):
                ent_layers.append(hidden_to_entropies(steps, mdl, layer=li))
                lp_layers.append(hidden_to_token_logprobs(steps, mdl, gen_ids, layer=li))
                
            if ent_layers == [] or lp_layers == []: # gaurd against no entropies/logprobs
                assert False, "No entropy or logprobs found."

            # Summarize historical data across layers
            # Updated from simple summaries per layer to across layers to be CHAIRS-style      
            T = len(last_lp_seq) # number of generated tokens
            chair_token_traces = []
            for t in range(T):
                ent_trace = [ent_layers[L][t] for L in range(len(ent_layers))]
                lp_trace = [lp_layers[L][t]  for L in range(len(lp_layers))]
                chair_token_traces.append({
                    "t": t,
                    "ent_summary": summarize_entropies(ent_trace), # {mean,std,min,max,slope}
                    "lp_summary": summarize_logprobs(lp_trace),
                })

            # Pick an answer based on log-likeihood and check if it's correct
            choices = ex["mc1_targets"]["choices"]
            labels = ex["mc1_targets"]["labels"]
            guess = pick_by_loglikelihood(mdl, tok, q, choices, device, policy="alpha", alpha=0.7)
            is_correct = (labels[guess] == 1)
            correct += int(is_correct)
            
            # Stable ID (prefer dataset's id; fallback to deterministic hash of question text)
            qid = ex.get("id", None)
            if qid is None:
                qid = f"tqa:{hashlib.sha1(q.encode()).hexdigest()[:12]}"

            split = split_bucket(str(qid))
            by_split[split]["n"] += 1
            by_split[split]["correct"] += int(is_correct)
            
            # Save full per-layer sequences (historical layers only, excluding final layer)
            historical_layers = {
                "layer_indices_in_range": list(range(hist_start, hist_end)),
                "logprobs_seq_by_layer": lp_layers,
                "entropies_seq_by_layer": ent_layers
            }

            # Store the results for jsonl
            rec = {
                "qid": qid,
                "split": split,
                "model_name": model,
                "question": q,
                "choices": choices,
                "labels": labels,
                "generated": gen_text[0],
                "pred_index": guess,
                "pred_text": choices[guess],
                "correct": is_correct,
                "hallucination": not is_correct,
                "num_layers_total": num_layers,
                "historical_layer_range": [hist_start, hist_end],
                "chair_token_traces": chair_token_traces,
                "historical_layers": historical_layers,
                "last_layer": {
                    "logprobs_seq": last_lp_seq,
                    "entropies_seq": last_ent_seq,
                },
            }

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Print out metrics and store in a json file
    metrics = {
        "timestamp": stamp,
        "dataset": "truthful_qa:multiple_choice",
        "n_total": len(ds),
        "overall_accuracy": correct / total if total else 0.0,
        "per_split": {
            s: {
                "n": v["n"],
                "accuracy": (v["correct"] / v["n"] if v["n"] else 0.0)
            }
            for s, v in by_split.items()
        },
        "model": model,
        "outname": outname,
        "predictions_file": str(pred_path),
    }
    with open(met_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    diagnostic_check()

if __name__ == "__main__":
    main()