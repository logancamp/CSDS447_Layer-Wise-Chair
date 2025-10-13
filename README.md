# CSDC447_Layer-Wise-Chair
Course project experimenting with an expanded CHAIR concept for layer wise hallucination detection
<br>
We'll probably want to set up docker or some more robust env setup
We need to expand the model to take layer history
We need to up the accuracy
We need aneasier way to run this, maybe a main file of some kind
We need to link other papers and datasets in here
We need to add comments and make it more readable
Reduce output files maybe or label them better even if just in here
Collect data as benchmarks (organize)
<br>
# BASH:
## 0) activate env
conda activate chair-lite
export TOKENIZERS_PARALLELISM=false
<br>
## 1) choose a model once for this session:
export CHAIR_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"   # ungated & fast
<br>
## 2) evaluate LLM on MC1 and store logits/probs for CHAIR training (takes a long time)
python src/eval_mc1.py --limit 500 --max_new_tokens 64 \
  --outname train_run --store_token_probs --token_prob_cap 64 \
  ${CHAIR_MODEL:+--model "$CHAIR_MODEL"}

### - tag responses as true/false & hallucination labels
python src/tag_mc1.py   --preds outputs/train_run.jsonl

### - convert tagged outputs into numerical feature vectors
python src/featurize.py --preds outputs/train_run.jsonl --K 32

### - train the CHAIR logistic-regression classifier
python src/train_chair.py --features outputs/train_run.features.csv --out outputs/chair_clf.pkl
<br>
## 3) collect a new MC1 sample for testing the CHAIR classifier (takes a long time)
python src/eval_mc1.py --limit 200 --max_new_tokens 64 \
  --outname eval_run --store_token_probs --token_prob_cap 64 \
  ${CHAIR_MODEL:+--model "$CHAIR_MODEL"}

### - apply trained CHAIR classifier to new eval set
python src/predict_chair.py --model_pkl outputs/chair_clf.pkl --preds outputs/eval_run.jsonl

### - generate markdown + CSV report of CHAIR predictions
python src/report.py       --preds outputs/eval_run.chair_scored.jsonl

### - aggregate all metrics across previous runs for comparison
python src/aggregate.py



