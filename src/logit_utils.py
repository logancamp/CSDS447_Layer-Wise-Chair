# src/logit_utils.py
import torch
import math

def scores_to_token_logprobs(scores, gen_token_ids):
    """
    Convert HuggingFace generate() scores -> log p(selected_token) per step.
    scores: list[Tensor] of shape [batch, vocab] for each generated step
    gen_token_ids: 1D LongTensor of generated token ids (continuation only, length == len(scores))
    Returns: 1D list[float] of log-probs (base-e) for each generated token.
    """
    logps = []
    for step_logits, tok_id in zip(scores, gen_token_ids):
        # step_logits: [1, vocab]
        lp = torch.log_softmax(step_logits[0], dim=-1)[tok_id.item()].item()
        logps.append(lp)
    return logps

def summarize_logprobs(logps):
    """
    Returns compact scalar stats for a sequence of per-token log-probs.
    """
    if not logps:
        return {"avg_logprob": 0.0, "avg_prob": 0.0, "perplexity": float("inf")}
    avg_lp = sum(logps) / len(logps)
    avg_p  = math.exp(avg_lp)
    ppl    = math.exp(-avg_lp)  # classic per-token perplexity
    return {"avg_logprob": avg_lp, "avg_prob": avg_p, "perplexity": ppl}

def step_entropy(logits_row):
    # logits_row: 1D tensor [vocab]; returns H(p) = -sum p log p
    import torch
    logp = torch.log_softmax(logits_row, dim=-1)
    p = torch.exp(logp)
    return float(-(p * logp).sum().item())

def scores_to_entropies(scores):
    # scores: list[Tensor [1,vocab]]
    return [step_entropy(s[0]) for s in scores]
