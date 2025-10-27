import argparse, json, time
from pathlib import Path
from datasets import load_dataset # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from logit_utils import summarize_logprobs, summarize_entropies, hidden_to_token_logprobs, hidden_to_entropies
from typing import cast
from transformers.generation.utils import GenerateDecoderOnlyOutput
import torch.nn.functional as F

"""
SUMMARY:
Evaluate a language model on TruthfulQA multiple-choice (mc1) questions.
Runs the model, gets the raw text output, picks a multiple-choice answer (via log-likelihood), and measures accuracy.
Stores raw results in a JSONL file, one entry per question, with various confidence metrics.
Later this data is tagged by tag_mc1 for training and pre-processed in predict_chair.
"""

# System prompt to use for all questions
SYS = "You are concise and truthful. Answer the question factually and directly."

def fetch_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outname", default="", 
        help="Optional base name for output files")
    ap.add_argument("--store_token_probs", action="store_true",
        help="Store per-token log-probs (truncated) and generated token ids")
    ap.add_argument("--token_prob_cap", type=int, default=32,
        help="Max number of token log-probs to store per example")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--outdir", default="outputs")
    return ap.parse_args()

# Return the index of the candidate with the highest log-likelihood: Same as TruthfulQA
def pick_by_loglikelihood(mdl, tok, question, choices, device):
    prompt_txt = f"{SYS}\nQ: {question}\nA: "
    prompt = tok(prompt_txt, return_tensors="pt")
    prompt_len = prompt["input_ids"].shape[1]

    scores = []
    for c in choices:
        # Encode prompt + choice together
        cp = tok(prompt_txt + c, return_tensors="pt")
        input_ids = cp["input_ids"].to(device)
        attn_mask = cp.get("attention_mask", None)
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)

        # Mask the prompt tokens so loss is only computed for choice
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100  # without prompt
        with torch.no_grad():
            out = mdl(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            num_labeled = (labels != -100).sum().item()
            loglik = -out.loss.item() * max(1, num_labeled)
        scores.append(loglik)
    return int(torch.tensor(scores).argmax().item())

def main():
    args = fetch_args()

    # Set up output paths
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_id = f"mc1_{Path(args.model).name}_{stamp}_n{args.limit}" # default run id
    if args.outname: # override run_id if an outname is given
        run_id = args.outname
    
    pred_path = outdir / f"{run_id}.jsonl"
    met_path = outdir / f"{run_id}.metrics.json"
    
    # Check if cuda is available for device
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    # Load model and tokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    
    # Ensure pad token exists (fixes a bug with some models)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    # Load the LLM model
    mdl = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map = "auto" if use_cuda else None,
        torch_dtype = torch.float16 if use_cuda else None,
    )
    mdl.to(device)
    mdl.eval() # Switch model to eval mode

    # Load the dataset (split, shuffled, sliced)
    ds = load_dataset("truthful_qa","multiple_choice")[args.split].shuffle(seed=args.seed)
    start = max(0, args.offset)
    end = start + args.limit if args.limit else len(ds)
    ds = ds.select(range(start, min(end, len(ds))))
    
    """ Each example in the dataset looks like this:
    {
    "question": "...",
    "mc1_targets": {
            "choices": ["A...", "B...", "C..."],
            "labels": [0, 1, 0]  # one correct answer
        }
    }
    """

    total = len(ds); correct = 0
    with open(pred_path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            
            # Prepare the prompt
            q = ex["question"]
            prompt = f"{SYS}\nQ: {q}\nA: "
            inputs = tok(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k,v in inputs.items()} if use_cuda else inputs # move to gpu if available
            
            # Generate output
            with torch.inference_mode():
                out = mdl.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    output_hidden_states=True,
                    output_scores=True,
                    return_dict_in_generate=True
                )

            
            # Process output
            out = cast(GenerateDecoderOnlyOutput, out)
            prompt_len = inputs["input_ids"].shape[1]
            gen_ids = out.sequences[:, prompt_len:]

            # zero-token guard with refreshed scores/steps
            if gen_ids.size(1) == 0:
                out = mdl.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max(1, args.max_new_tokens),
                    min_new_tokens=1,
                    output_hidden_states=True,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
                gen_ids = out.sequences[:, prompt_len:]
                if gen_ids.size(1) == 0:
                    continue

            scores = out.scores
            steps  = out.hidden_states
            assert steps is not None, "Hidden states missing. Did you set output_hidden_states=True?"
            
            gen_text = tok.batch_decode(gen_ids, skip_special_tokens=True)

            # Last layer only: entropies and logprobs
            last_logp = []
            last_ent  = []
            for t, logits in enumerate(scores):
                row = logits[0].to(torch.float32)
                logp = torch.log_softmax(row, dim=-1)
                last_logp.append(logp[gen_ids[0, t]].item())
                p = torch.exp(logp)
                last_ent.append(float(-(p * logp).sum().item()))

            # Labels same shape as input, mask the prompt
            lbl = out.sequences.clone()
            lbl[:, :prompt_len] = -100
            attn = torch.ones_like(out.sequences, device=out.sequences.device)

            with torch.no_grad():
                chk = mdl(input_ids=out.sequences, attention_mask=attn, labels=lbl)

            num_labeled = (lbl != -100).sum().item()

            # Compute CE over next-token logits to match model's criterion exactly
            approx_ce = 0.0
            for t, logits in enumerate(scores):  # logits: [B, V]
                approx_ce += F.cross_entropy(logits, gen_ids[:, t], reduction="sum").item()

            # Allow small fp drift
            assert abs(chk.loss.item() * num_labeled - approx_ce) < 1e-1, "Logprob alignment mismatch"

            # Historical: all layer entropies and logprobs
            num_layers = len(steps[0])
            hist_start, hist_end = 1, num_layers - 1
            
            layer_data = []
            for li in range(hist_start, hist_end): # skip final layer and embedding layer
                ent = hidden_to_entropies(steps, mdl, layer=li)
                lp  = hidden_to_token_logprobs(steps, mdl, gen_ids, layer=li)
                layer_data.append((ent, lp))

            # Summarize historical data per layer
            hist_entropies = []
            hist_logprobs = []
            for thing in layer_data:
                entropies, logprobs = thing
                hist_entropies.append(summarize_entropies(entropies))
                hist_logprobs.append(summarize_logprobs(logprobs))

            # Pick an answer based on log-likeihood and check if it's correct
            choices = ex["mc1_targets"]["choices"]
            labels = ex["mc1_targets"]["labels"]
            guess = pick_by_loglikelihood(mdl, tok, q, choices, device)
            is_correct = (labels[guess] == 1)
            correct += int(is_correct)
            
            # Add an id to make sure each question is uniquely identified
            qid = ex.get("id", None)
            if qid is None:
                qid = f"{args.split}:{start + i}"  # n = 0,1,2... as you iterate

            # Store the results for jsonl
            rec = {
                "qid": qid,
                "split": args.split,
                "model_name": args.model,
                "question": q,
                "choices": choices,
                "labels": labels,
                "generated": gen_text[0],
                "pred_index": guess,
                "pred_text": choices[guess],
                "correct": bool(is_correct),
                "num_layers_total": num_layers,
                "historical_layer_range": [hist_start, hist_end],
            }
            
            # Create json for historical layer summaries
            historical_layers = [
                {
                    "layer": i+1,
                    "ent_summary": ent,
                    "lp_summary": lp
                }
                for i, (ent, lp) in enumerate(zip(hist_entropies, hist_logprobs))
            ]
            rec["historical_layers"] = historical_layers

            # Create json for last layer info
            rec["last_layer"] = {
                    "logprobs_seq": last_logp,
                    "entropies_seq": last_ent,
                }

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Print out metrics and store in a json file
    acc = correct / total if total else 0.0
    metrics = {
        "dataset": "truthful_qa:multiple_choice",
        "n": total,
        "accuracy": acc,
        "error_rate": 1.0 - acc,
        "model": args.model,
        "run_id": run_id,
        "predictions_file": str(pred_path),
    }
    with open(met_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
