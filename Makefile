# --- Setup ---
setup:
	export TOKENIZERS_PARALLELISM=false
	mkdir -p outputs

# --- Data Collection (Shared) ---
collect_train_data: setup
	python src/eval_mc1.py --seed 42 --train True

collect_test_data: setup
	python src/eval_mc1.py --seed 42 --train False

# --- Tagging (Shared) ---
tag_train_data:
	python src/tag_mc1.py --preds outputs/train_run.jsonl

tag_test_data:
	python src/tag_mc1.py --preds outputs/eval_run.jsonl


# --- V1: Logistic Regression Pipeline ---

featurize_train_data_lr:
	python src/featurize.py --preds outputs/train_run.tagged.jsonl --K 18 \
	  --out outputs/train_run.features.csv

train_model_lr: featurize_train_data_lr
	python src/train_chair.py --features outputs/train_run.features.csv \
	  --out outputs/chair_clf.pkl

featurize_test_data_lr:
	python src/featurize.py --preds outputs/eval_run.tagged.jsonl --K 18 \
	  --out outputs/eval_run.features.csv

predict_test_data_lr: featurize_test_data_lr
	python src/predict_chair.py \
	  --model_pkl outputs/chair_clf.pkl \
	  --preds_jsonl outputs/eval_run.jsonl \
	  --features_csv outputs/eval_run.features.csv

full_run_lr: setup collect_train_data tag_train_data train_model_lr collect_test_data tag_test_data predict_test_data_lr


# --- V2: Neural Network (NN) Pipeline ---

featurize_train_data_nn:
	python src/featurize_nn.py --preds outputs/train_run.tagged.jsonl --K 32 \
	  --out outputs/train_run.features.jsonl

train_model_nn: featurize_train_data_nn
	python src/train_chair_nn.py --features outputs/train_run.features.jsonl \
	  --out outputs/chair_nn.pth --epochs 10

featurize_test_data_nn:
	python src/featurize_nn.py --preds outputs/eval_run.tagged.jsonl --K 32 \
	  --out outputs/eval_run.features.jsonl

predict_test_data_nn: featurize_test_data_nn
	python src/predict_chair_nn.py \
	  --model_path outputs/chair_nn.pth \
	  --preds_jsonl outputs/eval_run.jsonl \
	  --features_jsonl outputs/eval_run.features.jsonl

full_run_nn: setup collect_train_data tag_train_data train_model_nn collect_test_data tag_test_data predict_test_data_nn


# --- Aliases ---
# Default 'train_model' and 'predict_test_data' to the original LR versions
# for backward compatibility.
train_model: train_model_lr
predict_test_data: predict_test_data_lr
full_run: full_run_lr

# You can run 'make full_run_nn' to run the new pipeline.
