from transformers import AutoConfig
import re
import pandas as pd

model = "meta-llama/Llama-3.2-1B-Instruct" # 16 15
# model = "meta-llama/Meta-Llama-3-8B-Instruct" # 32 31
# model = "Qwen/Qwen3-4B-Instruct-2507" # 36 35
# model = "Qwen/Qwen3-4B-Thinking-2507" # 36 35
# model = "Qwen/Qwen3-8B" # 36 35
# model = "mistralai/Ministral-8B-Instruct-2410" #36 35

# model size
cfg = AutoConfig.from_pretrained(model)
print("num_hidden_layers:", getattr(cfg, "num_hidden_layers", None))

# used size
df = pd.read_csv("data/L1b/mc1_results.historical_layers.csv")  # adjust path/name

layer_ids = set()
for col in df.columns:
    # tweak regex if your naming is slightly different
    m = re.search(r"L?(\d+)", col)   # matches 'L3' or 'L_3'
    if m:
        layer_ids.add(int(m.group(1)))

layer_ids = sorted(layer_ids)
print("Layer IDs present in features:", layer_ids)

if layer_ids:
    print("Min layer:", layer_ids[0])
    print("Max layer:", layer_ids[-1])
    print("Count of distinct layers:", len(layer_ids))