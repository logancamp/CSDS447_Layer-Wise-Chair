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

sum_hist:
	python src/summarize_hist.py \
		--in_csv outputs/mc1_results.historical_layers.csv \
		--out_csv outputs//mc1_results.features.csv \
		--omit_last_k_layers ${K}

set_layer: featurize_data sum_hist

convert:
	python src/convert.py --features outputs/mc1_results.features.csv --out chair_data


# 0 = n, 1 = n-1 etc for layerwise
# K ?= 0

# --- V1: Logistic Regression Pipeline ---
train_model_lr:
	python src/train_chair_lr.py --features chair_data/chair_features.csv

predict_lr:
	python src/predict_chair_lr.py \
	  --model_pkl outputs/chair_classifier_lr.pkl \
	  --train_metrics outputs/chair_classifier_lr.train_metrics.json \
	  --test_data outputs/mc1_results.jsonl \
	  --features chair_data/chair_features.csv

output_run_lr: set_layer convert train_model_lr
test_run_lr: set_layer convert predict_lr


# 0 = n, 1 = n-1 etc for layerwise
K ?= 0

# --- V2: Neural Network (NN) Pipeline ---
train_model_nn:
	python src/train_chair_nn.py --features chair_data/chair_features.csv --epochs 10

predict_nn:
	python src/predict_chair_nn.py \
	  --model_pth outputs/chair_classifier_nn.pth \
	  --test_data outputs/mc1_results.jsonl \
	  --features chair_data/chair_features.csv

output_run_nn: set_layer convert train_model_nn
test_run_nn: set_layer convert predict_nn


# --- Hyperparameter Tuning For NN (not-effective) ---
tune: 
	python src/tune_chair.py