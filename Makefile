# --- Setup ---
setup:
	export TOKENIZERS_PARALLELISM=false
	mkdir -p outputs


# --- Data Collection (Shared) ---
collect_data:
	python src/eval_mc1.py --seed 42


# --- V1: Logistic Regression Pipeline ---
featurize_data:
	python src/featurize.py --tmodel lr --preds outputs/mc1_results.jsonl --K 18

train_model_lr:
	python src/train_chair.py --features outputs/mc1_results.features.csv

predict_lr:
	python src/predict_chair.py \
	  --model_pkl outputs/chair_classifier.pkl \
	  --train_metrics outputs/chair_classifier.train_metrics.json \
	  --test_data outputs/mc1_results.jsonl \
	  --features outputs/mc1_results.features.csv

full_run_lr: setup collect_data featurize_data train_model_lr predict_lr


# --- V2: Neural Network (NN) Pipeline ---
train_model_nn:
	python src/train_chair_nn.py --features outputs/mc1_results.features.csv --epochs 10

predict_nn:
	python src/predict_chair_nn.py \
	  --model_pth outputs/chair_nn.pth \
	  --test_data outputs/mc1_results.jsonl \
	  --features outputs/mc1_results.features.csv

full_run_nn: setup collect_data featurize_data train_model_nn predict_nn


# --- Aliases ---
# Default 'train_model' and 'predict_test_data' to the original LR versions
# for backward compatibility.
train_model: train_model_lr
predict_test_data: predict_test_data_lr
full_run: full_run_lr

# You can run 'make full_run_nn' to run the new pipeline.
