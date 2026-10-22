#!/usr/bin/env bash
set -euo pipefail
cd /workspace/pose_reconstract
export UV_PYTHON_INSTALL_DIR="$PWD/.python"
export UV_CACHE_DIR="$PWD/.uv-cache"
export UV_LINK_MODE=hardlink
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4
UV=/root/.local/bin/uv
mkdir -p artifacts/bootstrap inputs repos
date -u +%FT%TZ
df -h .
if [ ! -x .venv/bin/python ]; then
  "$UV" venv --python 3.10 .venv
fi
"$UV" pip install --python .venv/bin/python --no-index --find-links wheels-linux \
  -r experiments/requirements-geometry.txt
.venv/bin/python scripts/logged_run.py .venv/bin/python scripts/run_geometry_experiments.py \
  --config experiments/geometry_v1.json --run-id cpu_geom_v1_001
/opt/simready/pamo-env/bin/python -m venv --system-site-packages .venv-gpu
"$UV" pip install --python .venv-gpu/bin/python --no-index --find-links wheels-linux \
  -r experiments/requirements-geometry.txt
.venv-gpu/bin/python scripts/logged_run.py .venv-gpu/bin/python scripts/gpu_smoke.py
"$UV" pip freeze --python .venv/bin/python > artifacts/bootstrap/freeze.txt
.venv-gpu/bin/python -m pip freeze > artifacts/bootstrap/freeze-gpu.txt
date -u +%FT%TZ
