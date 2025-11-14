# CSDS447_Layer-Wise-Chair
Course project experimenting with an expanded CHAIR concept for layer wise hallucination detection<br>
Lower Chair Score = Higher Predicted Chance of Hallucination
<br><br>

## Models and Datasets:
CHAIR: https://github.com/eggachecat/CHAIR <br>
* Note: Due to bad documentation, seemingly broken code, and hardware limitations early on, we opted to train our own Logarithmic Regression model. We planned to update this to a similar attention model for Demo 2 but the model create performed very poorly, we hope to improve this model as well as the Logarithmic Regression to get better detection results and use the best model and LLM for our final layerwise analysis. Potentially transition to a different model and LLM if needed (see bellow for plans going forward). *
<br><br>

LLAMA 8B: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct <br>
LLAMA 1B: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct <br>
TinyLLAMA (initial fast testing): https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0 <br>
<br>

TruthfulQA: https://github.com/sylinrl/TruthfulQA <br>
MMLU: https://github.com/hendrycks/test?tab=readme-ov-file
<br><br>

## Baseline + Methodology
Our benchmarks are the hallucination detection outputs. We detect these by comparing TruthfulQA multiple choice responses with generated outputs. Currently this is a log-likelihood function the same as TruthfulQA.

## Goal:
Are we able to detect these hallucinations earlier rather than only in the last layer as CHAIR and other detection methods employ?
<br><br>

## Architecture:
- 70/20/10 split training/validation/test data
- Training data consists of across layer summary stats + last layer token logprobs and entropies
- Cross-layer stats include: mean, max, min, standar deviation, and slope
- Simple logistic regression and attention network options currently exist for analysis
<br><br>

## Limitations for Future Work:
- Highly skewed data for Llama models
- Models consistently hallucinate, not enough correct data (20%)
  - In outputs, lr models are adjusted to prevent threshold tuning from learning to only predict YES to hallucination
  - In outputs, nn models still use threshold tuneing but appear to only predict YES to hallucinations
  - lr models predict similarly to nn when threshold tuning remains in place

### Progress/Attempts for Limitation Fixes (attempts in LR model):
- Re-weighing minority features
- Down-sampling majority samples
- Use of a tree model over logarithmic regression
- SMOTE for synthetic data

## Goals and Next Steps:
- Next we hope to find, or generate more robust datasets or a better model:
  - Find available data from numerous other sources if they exist (Potenitally non-truthfulQA data)
  - Try merging multiple TQA styled baselines (more correct responses, not garenteed)
  - Try to generate with different LLM models (not garenteed)
  - Potentially find alternative prediction model approaches like SelfCheck grey box methods, or other models similar to Chair for classification if model outputs are unuseable but we hope to improve the current models first to prevent large scale refactors
- After,
  - We will adjust our dataset to include subsets for partially completed layer data (i.e. data up through k layers instead of the full layer data)
  - Or collect per layer data and train per layer to learn internal patterns. (most likely this one due to limited data availability) <br>
- Lastly, we will run classification at each layer to track hallucinations throughout the process of generation for better explainability for LLM hallucination early detection. We hope to see significant changes in accuracy metrics to pinpoint which layers are the most important attributes for hallucination results. (Foe example: Do the last four layers, if omited, make a difference in prediction outputs?)
<br><br>

## AI Use Disclosure:
Generative Ai models (GPT-5) was used to help with debugging, and some code generation, primarily file writing tasks. It was also used to help with tuning and ideation for data pre-processing to improve outputs or to learn about library/model implimentation steps. Outputs were both cross checked with online forums like Stackoverflow or GeeksForGeeks and official documentations, as well as edited and cleaned/fixed from personal knowledge. AI tools like copilot were also used occasionally for debugging tasks within the codebase using VSCode's "fix" tool, also verified and fixed using personal knowledge.

### General Prompt Formats:
"*{context like code snippets, occasionally used for specific tasks}*<br>
Tell me how to impliment a *{model/outputting task}* using *{library like sklearn}*"
<br><br>
"Explain how a general implimentation of *{model}* from sklearn would be coded. What packages are needed?"
<br><br>
"*{Error Context}*<br>
Where does this error occur in my implimentation? <br>
*{Code Snippet Context}*"
<br><br>
"Explain: *{Context, typically statistics or coding errors}*"
<br><br><br>

# Get Started (BASH):
## 0) Setup Dependencies
`sudo singularity build singularity.sif container.def`

## 1) Activate + Setup Environment
`singularity shell --nv singularity.sif` <br>

singularity shell --nv --bind $PWD:/workspace --pwd /workspace new_singularity.sif

`make setup`

## 2) Data Accumulation
`make collect_data`
<br><br>

# Running Full Logarithmic Regression (BASH):
## 0) Run the full data collection + training + prediction:
`make full_run_lr`

# Running Full Attention Neural Network (BASH):
## 0) Run the full data collection + training + prediction:
`make full_run_nn`
<br><br>

# Step-by-step Run LR/NN (BASH):
## 1) Featurize the train data - numeric:
`make featurize_data_lr` <br>
`make featurize_data_nn`

## 2) Trained Model + Val + Test:
`make train_model_lr` <br>
`make train_model_nn`

## Run Prediction on a Model (Independent-Post Training):
`make predict_lr` <br>
`make predict_nn`
<br><br><br>

# FILES:
## raw, human readile data: 
- mc1_results.jsonl

## data collection result metrics (how the model did):
- mc1_results.metrics.json

## feature, test and train data in numerical form:
- mc1_results.features.csv

## actual usable trained model:
- chair_classifier_lr.pkl
- chair_classifier_nn.pth

## train and prediction metrics (train, val, and test also has col order and threshold):
- chair_classifier_lr.train_metrics.json
- chair_classifier_nn.predict_metrics.json

## prediction results simplified:
- mc1_results.lr_chair_scores.csv
- mc1_results.nn_chair_scores.csv
