setup:
    export TOKENIZERS_PARALLELISM=false
    export CHAIR_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"


collect_train_data:
    python src/eval_mc1.py --limit 500 --max_new_tokens 64 \
        --outname train_run --store_token_probs --token_prob_cap 64 \
        ${CHAIR_MODEL:+--model "$CHAIR_MODEL"}


tag_train_data:
    python src/tag_mc1.py   --preds outputs/train_run.jsonl


featurize_train_data:
    python src/featurize.py --preds outputs/train_run.jsonl --K 32


train_model:
    python src/train_chair.py --features outputs/train_run.features.csv --out outputs/chair_clf.pkl


collect_test_data:
    python src/eval_mc1.py --limit 200 --max_new_tokens 64 \
        --outname eval_run --store_token_probs --token_prob_cap 64 \
        ${CHAIR_MODEL:+--model "$CHAIR_MODEL"}


predict_test_data:
    python src/predict_chair.py --model_pkl outputs/chair_clf.pkl --preds outputs/eval_run.jsonl


full_run: setup collect_train_data tag_train_data featurize_train_data train_model collect_test_data predict_test_data


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
