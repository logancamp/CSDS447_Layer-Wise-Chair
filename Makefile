# --- Setup ---
setup:
	export TOKENIZERS_PARALLELISM=false
	mkdir -p outputs


# --- Data Collection (Shared) ---
collect_data:
	python src/eval_mc1.py --seed 42


# --- Feature Extraction (Shared) ---
featurize_data:
	python src/featurize.py --preds outputs/mc1_results.jsonl --K 18
	python src/featurize_hist.py --preds outputs/mc1_results.jsonl --K 18


# --- V1: Logistic Regression Pipeline ---
train_model_lr:
	python src/train_chair_lr.py --features outputs/mc1_results.features.csv

predict_lr:
	python src/predict_chair_lr.py \
	  --model_pkl outputs/chair_classifier_lr.pkl \
	  --train_metrics outputs/chair_classifier_lr.train_metrics.json \
	  --test_data outputs/mc1_results.jsonl \
	  --features outputs/mc1_results.features.csv

output_run_lr: featurize_data train_model_lr
full_run_lr: setup collect_data featurize_data train_model_lr predict_lr


# --- V2: Neural Network (NN) Pipeline ---
train_model_nn:
	python src/train_chair_nn.py --features outputs/mc1_results.features.csv --epochs 10

predict_nn:
	python src/predict_chair_nn.py \
	  --model_pth outputs/chair_classifier_nn.pth \
	  --test_data outputs/mc1_results.jsonl \
	  --features outputs/mc1_results.features.csv

output_run_nn: featurize_data train_model_nn
full_run_nn: setup collect_data featurize_data train_model_nn predict_nn


# --- Aliases ---
# Default 'train_model' and 'predict_test_data' to the original LR versions
# for backward compatibility.
train_model: train_model_lr
predict_test_data: predict_test_data_lr
full_run: full_run_lr

# You can run 'make full_run_nn' to run the new pipeline.
