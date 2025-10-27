import torch
import math

# Internal helper: get logits from hidden states at last position
def _logits_from_hidden_lastpos(h_lastpos, mdl):
    x = mdl.model.norm(h_lastpos)
    logits = mdl.lm_head(x)
    return logits

# Helper to compute entropy from logits row
def step_entropy(logits_row):
    logp = torch.log_softmax(logits_row, dim=-1)
    p = torch.exp(logp)
    return float(-(p * logp).sum().item())

# Returns list[float] of per-token log-probs from a given layer
def hidden_to_token_logprobs(steps, mdl, gen_ids, layer=-1):
    assert gen_ids.shape[0] == 1, "utils assume batch size 1"
    logps = []
    for t, step in enumerate(steps):
        y_t = gen_ids[:, t]               # [1]
        h_last = step[layer][:, -1, :].float()
        logits = _logits_from_hidden_lastpos(h_last, mdl)[0].float()
        lp_t = torch.log_softmax(logits, dim=-1)[y_t.item()].item()
        logps.append(lp_t)
    return logps

# Returns compact scalar stats for a sequence of per-token log-probs
def summarize_logprobs(logps):
    if not logps:
        return {"avg_logprob": 0.0, "avg_prob": 0.0, "perplexity": float("inf")}
    avg_lp = sum(logps) / len(logps)
    avg_p  = math.exp(avg_lp)
    ppl = math.exp(-avg_lp)
    return {"avg_logprob": avg_lp, "avg_prob": avg_p, "perplexity": ppl}

# Returns compact scalar stats for a sequence of entropies
def summarize_entropies(xs):
    if not xs: 
        return {"mean":0.0,"std":0.0,"min":0.0,"max":0.0,"first":0.0,"last":0.0,"delta":0.0}
    m = sum(xs)/len(xs)
    v = sum((x-m)**2 for x in xs)/len(xs)
    return {"mean":m,"std":v**0.5,"min":min(xs),"max":max(xs),"first":xs[0],"last":xs[-1],"delta":xs[-1]-xs[0]}

# Returns list[float] of per-token entropies from a given layer
def hidden_to_entropies(steps, mdl, layer=-1):
    ents = []
    for step in steps:
        h_last = step[layer][:, -1, :]
        logits = _logits_from_hidden_lastpos(h_last, mdl)[0]
        ents.append(step_entropy(logits))
    return ents