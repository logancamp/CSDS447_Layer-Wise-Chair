# CSDS447_Layer-Wise-Chair
Course project experimenting with an expanded CHAIR concept for layer wise hallucination detection<br>
Lower Chair Score = Higher Predicted Chance of Hallucination

<br><br>

## Models and Datasets:
CHAIR: https://github.com/eggachecat/CHAIR <br>
* Note: Due to bad documentation, seemingly broken code, and hardware limitations early on, we opted to train our own Logarithmic Regression model. We planned to update this to a similar attention model for Demo 2 but the model create performed very poorly, we hope to improve this model as well as the Logarithmic Regression to get better detection results and use the best model and LLM for our final layerwise analysis. Potentially transition to a different model and LLM if needed (see bellow for plans going forward). * <br>

TruthfulQA: https://github.com/sylinrl/TruthfulQA <br>

LLAMA 8B: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct <br>
LLAMA 1B: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct <br>
TinyLLAMA (initial fast testing): https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0 <br>
Qwen 4B: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 <br>
Qwen 4b Think: https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507 <br>
Qwen 8b: https://huggingface.co/Qwen/Qwen3-8B <br>
Mistral 8b: https://huggingface.co/mistralai/Ministral-8B-Instruct-2410 <br>

<br><br>

## Baseline + Methodology
Our benchmarks are the hallucination detection outputs. We detect these by comparing TruthfulQA multiple choice responses with generated outputs. Currently this is a log-likelihood function the same as TruthfulQA.

### Goal:
Are we able to detect these hallucinations earlier rather than only in the last layer as CHAIR and other detection methods employ?

<br><br>

## Architecture:
- 80/10/10 split training/validation/test data
- Training data consists of across layer summary stats + last layer token logprobs and entropy summaries
- Cross-layer stats include: mean, max, min, standar deviation, and slope
- Simple logistic regression and attention network options currently exist for analysis

<br><br>

## Limitations for Future Work:
- Highly skewed data for all models
- Models consistently hallucinate, not enough correct data (20%)

### Accounting for Limitations:
- Re-weighing minority features
- Down-sampling majority samples
- Use of a tree model over logarithmic regression
- SMOTE for synthetic data

<br> The best performing patterns were used for each model individually but remained consistent for layerwise analysis. Originally our data was resulting in highly volitile results, subject to high variation given seed changes. We started by refactoring the data features used. Upon further investigation our data schema was somewhat different than the original CHAIR intentions, with this we created a convert file to adapt the current data into a better schema. The overall representation was similar but we made the following changes:
- Each layer is represented in the input by a token-aggregated scalar (mean logprob/entropy over the tail tokens), not raw per-token values.
- The historical summary features then take these per-layer scalars and compute cross-layer statistics (mean, std, min, max, slope, etc.) over the prefix of layers 0…N−k, which acts like an n-gram–style offset across runs.
- For any given k, the “final layer” features (last_lp_tail_t, last_ent_tail_t) are simply the per-layer token-mean values from the effective last layer (N−k), unsummarized across layers.

<br><br>

## Experimentation:
After altering the data implimentation, we tested the top performing models and dataset from our benchmark data. This benchmarking was simply the full layer dataset with k=0. <br>
- For each dataset, on their respective best model implimentation, we trained the given model per layer using a converted dataset with offset k for n-k layers for the last 6 layers and first 1, 5, and 10 layers. So, each of the 6 final layers and 1, 5, and 10 first layers data are tested via re-training the model on these individual layers.
- Then, in addition, we tested these same layers for the top 2 performing baselines in each model without retraining. So, for this test we used a model trained on the baseline, k=0, and conducted the same prediction over the test set for a given layer.

### Performance Outcomes:
We saw average, somewhat above random results for all benchmarks and analyzed the change in performance across layers, generally we saw diminishing returns as layers got older but some layers seem to perform on par with baselines. Further analysis will be listed in the report and excel file attacked in outputs/demo3_excel...

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
Note: make sure to add the 2 json files from data into the outputs folder before running: output/mc1_results.json and output/mc1_results.metrics.json. You can also add the two feature csv's but thesse will be computed for you upon running.
## 0) Setup Dependencies
`sudo singularity build singularity.sif container.def`

## 1) Activate + Setup Environment
`singularity shell --nv singularity.sif` <br>
`singularity shell --nv --bind $PWD:/workspace --pwd /workspace new_singularity.sif` <br>
`make setup`

## 2) Data Accumulation
`make collect_data`
<br><br>

# Running Full Logarithmic Regression (BASH):
note: set K in makefile to a designated offset, k=0 for baseline
## 0) Run the full data conversion + training + prediction:
`make output_run_lr` <br>
## 0) Run the full data conversion + prediction (no retraining):
`make output_run_lr` start with k=0 to train a model for prediction <br>
`make test_run_lr`

# Running Full Attention Neural Network (BASH):
## 0) Run the full data conversion + training + prediction:
`make output_run_nn` <br>
## 0) Run the full data conversion + prediction (no retraining):
`make output_run_nn` start with k=0 to train a model for prediction <br>
`make test_run_nn`

<br><br><br>

# FILES:
## raw, human readile data: 
- mc1_results.jsonl

## data collection result metrics (how the model did):
- mc1_results.metrics.json

## feature, test and train data in numerical form:
- mc1_results.features.csv
- mc1_results.historical_layers.csv

## actual training data being used:
- chair_data/chair_features.csv

## actual usable trained model:
- chair_classifier_lr.pkl
- chair_classifier_nn.pth

## train and prediction metrics (train, val, and test also has col order and threshold):
- chair_classifier_lr.train_metrics.json
- chair_classifier_lr.predict_metrics.json
- chair_classifier_nn.metrics.json
- chair_classifier_nn.predict_metrics.json
