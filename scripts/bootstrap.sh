#!/usr/bin/env bash
set -euo pipefail
ENV=${1:-chair-lite}
conda create -y -n "$ENV" python=3.10
conda activate "$ENV"
conda install -y pytorch=2.3.1 cpuonly -c pytorch -c conda-forge
pip install -r requirements.txt
python - <<'PY'
import torch; print("torch", torch.__version__, "cuda?", torch.cuda.is_available())
PY
