# CSDC447_Layer-Wise-Chair
Course project experimenting with an expanded CHAIR concept for layer wise hallucination detection

### One-time setup
##### BASH:
conda activate chair-lite
huggingface-cli login   # if using gated Llama models

### collect train data (features from Llama)
##### BASH:
python src/eval_mc1.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --limit 500 --max_new_tokens 64 --outname train_llama3 \
  --store_token_probs --token_prob_cap 64 \
  --hidden_last_layers 3 --hidden_tail 32
  
Writes: outputs/train_llama3.jsonl, outputs/train_llama3.metrics.json

### Tag correctness (creates hallucination label)
##### BASH:
python src/tag_mc1.py --preds outputs/train_llama3.jsonl

Writes: outputs/train_llama3.tagged.jsonl, outputs/train_llama3.tagged.csv

### Featurize for CHAIR classifier
##### BASH:
python src/featurize.py --preds outputs/train_llama3.jsonl --K 32

Writes: outputs/train_llama3.features.csv

### Train CHAIR classifier (logistic regression)
##### BASH:
python src/train_chair.py \
  --features outputs/train_llama3.features.csv \
  --out outputs/chair_clf.pkl

Outputs: AUC/AP report (stdout), outputs/chair_clf.pkl

### Collect TEST data (same flags/schema)
##### BASH:
python src/eval_mc1.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --limit 200 --max_new_tokens 64 --outname eval_llama3 \
  --store_token_probs --token_prob_cap 64 \
  --hidden_last_layers 3 --hidden_tail 32

Writes: outputs/eval_llama3.jsonl, outputs/eval_llama3.metrics.json

### Predict hallucination on TEST (CHAIR scoring)
##### BASH:
python src/predict_chair.py \
  --model_pkl outputs/chair_clf.pkl \
  --preds outputs/eval_llama3.jsonl

Writes: outputs/eval_llama3.chair_scored.jsonl (adds chair_score ∈ [0,1])

### Report and aggregate
##### BASH:
# Per-run human report
python src/report.py --preds outputs/eval_llama3.chair_scored.jsonl
# Multi-run table (reads *.metrics.json)
python src/aggregate.py

Writes: outputs/eval_llama3.chair_scored.tagged.report.md and .report.metrics.json; aggregation prints to stdout


##Makefile shortcuts (optional):
##### BASH:
make chair_collect_train
make chair_train
make chair_score
