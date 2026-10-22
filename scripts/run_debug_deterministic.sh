#!/usr/bin/env bash
set -euo pipefail
cd /workspace/pose_reconstract
export CUDA_VISIBLE_DEVICES=1
export DEBUG_SERVER_URL=http://127.0.0.1:17777/event
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
.venv-gpu/bin/python scripts/logged_run.py .venv-gpu/bin/python scripts/train_rhd_supervised.py --config experiments/rhd_supervised_v4.json --run-id rhd_v4_deterministic_a --deterministic
.venv-gpu/bin/python scripts/logged_run.py .venv-gpu/bin/python scripts/train_rhd_supervised.py --config experiments/rhd_supervised_v4.json --run-id rhd_v4_deterministic_b --deterministic
.venv-gpu/bin/python scripts/logged_run.py .venv-gpu/bin/python scripts/train_rhd_supervised.py --config experiments/rhd_supervised_v5.json --run-id rhd_v5_deterministic --deterministic
