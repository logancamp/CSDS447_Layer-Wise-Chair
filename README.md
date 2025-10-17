# CSDS447_Layer-Wise-Chair
Course project experimenting with an expanded CHAIR concept for layer wise hallucination detection<br>
<br>

## Models and Datasets:
CHAIR: https://github.com/eggachecat/CHAIR <br>
<br>
LLAMA 8B: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct <br>
LLAMA 1B: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct <br>
TinyLLAMA (current): https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0 <br>
<br>
TruthfulQA: https://github.com/sylinrl/TruthfulQA <br>
MMLU: https://github.com/hendrycks/test?tab=readme-ov-file <br>

## Baseline + Methodology
Our benchmarks are the last layer detection outputs. Later (depending on resources) these will be upgraded to include layer history. We detect these by comparing TruthfulQA multiple choice responses with generated outputs. Currently this is a log-likelihood function the same as TruthfulQA.

## Goal:
Are we able to detect these hallucinations earlier rather than only in the last layer as CHAIR and other detection methods employ?

## Architecture:
- 80:20 split training/test data
- Training data consists of last layer summary stats + token logprobs
- Simple logistic regression currently <br><br>
- Later we will upgrade the model to an Attention + Feedforward network like CHAIR with a more robust model from above and include layer summaries for better baselines.
- After we will adjust our dataset to include partially completed layer data - or train per layer to learn internal patterns. If time and space allow we will try to do most or all layers, otherwise only the last 5-10. <br>
- Lastly we will attempt classification at each layer to track hallucinations throughout the process of generation for better explainability for LLM hallucination early detection.
<br>

# Get Started (BASH):
## 0) Setup Dependencies
sudo singularity build singularity.sif container.def

## 1) Activate + Setup Environment
singularity shell --nv singularity.sif <br>
makefile setup <br>

## 2) Run the full data collection + training + prediction:
makefile full_run <br><br>

# Step-by-step Run (BASH):
## 1) Collect training data through TruthfulQA:
makefile collect_train_data <br>

## 2) Tag the train data - human readible:
makefile tag_train_data <br>

## 3) Featurize the train data - numeric:
makefile featurize_train_data <br>

## 4) Collect testing data through TruthfulQA:
makefile collect_test_data <br>

## 5) Run chair-lite prediction from trained model:
makefile predict_test_data
