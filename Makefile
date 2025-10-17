setup:
	export TOKENIZERS_PARALLELISM=false
	export CHAIR_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"

collect_train_data:
	python src/eval_mc1.py --split validation --limit 500 --max_new_tokens 64 \
	  --outname train_run --store_token_probs --token_prob_cap 64 \
	  ${CHAIR_MODEL:+--model "$CHAIR_MODEL"}

tag_train_data:
	python src/tag_mc1.py   --preds outputs/train_run.jsonl

featurize_train_data:
	python src/featurize.py --preds outputs/train_run.jsonl --K 32

train_model:
	python src/train_chair.py --features outputs/train_run.features.csv --out outputs/chair_clf.pkl

collect_test_data:
	python src/eval_mc1.py --split validation --limit 200 --max_new_tokens 64 \
	  --outname eval_run --store_token_probs --token_prob_cap 64 \
	  ${CHAIR_MODEL:+--model "$CHAIR_MODEL"}

predict_test_data:
	python src/predict_chair.py --model_pkl outputs/chair_clf.pkl --preds outputs/eval_run.jsonl

full_run: setup collect_train_data tag_train_data featurize_train_data train_model collect_test_data predict_test_data