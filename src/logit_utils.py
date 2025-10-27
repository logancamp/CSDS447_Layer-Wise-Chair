import torch
import math

def _logits_from_hidden_lastpos(h_lastpos, mdl):
    # Cast input to the same dtype as lm_head weights (e.g., fp16 on CUDA)
    x = mdl.model.norm(h_lastpos).to(mdl.lm_head.weight.dtype)
    return mdl.lm_head(x)

def step_entropy(logits_row):
    # Do numerics in float32
    logp = torch.log_softmax(logits_row.to(torch.float32), dim=-1)
    p = torch.exp(logp)
    return float(-(p * logp).sum().item())

def hidden_to_token_logprobs(steps, mdl, gen_ids, layer=-1):
    logps = []
    for t, step in enumerate(steps):
        y_t = gen_ids[:, t]
        h_last = step[layer][:, -1, :]                       # [1, D], same dtype as model
        logits = _logits_from_hidden_lastpos(h_last, mdl)[0] # [V], dtype matches lm_head
        lp = torch.log_softmax(logits.to(torch.float32), dim=-1)[y_t.item()].item()
        logps.append(lp)
    return logps

def summarize_logprobs(logps):
    if not logps:
        return {"avg_logprob": 0.0, "avg_prob": 0.0, "perplexity": float("inf")}
    avg_lp = sum(logps) / len(logps)
    return {"avg_logprob": avg_lp, "avg_prob": math.exp(avg_lp), "perplexity": math.exp(-avg_lp)}

def summarize_entropies(xs):
    if not xs:
        return {"mean":0.0,"std":0.0,"min":0.0,"max":0.0,"first":0.0,"last":0.0,"delta":0.0}
    m = sum(xs)/len(xs)
    v = sum((x-m)**2 for x in xs)/len(xs)
    return {"mean":m,"std":v**0.5,"min":min(xs),"max":max(xs),"first":xs[0],"last":xs[-1],"delta":xs[-1]-xs[0]}

def hidden_to_entropies(steps, mdl, layer=-1):
    ents = []
    for step in steps:
        h_last = step[layer][:, -1, :]
        logits = _logits_from_hidden_lastpos(h_last, mdl)[0]
        ents.append(step_entropy(logits))
    return ents