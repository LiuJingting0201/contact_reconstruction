#!/usr/bin/env bash
set -euo pipefail
cd /workspace/pose_reconstract
export CUDA_VISIBLE_DEVICES=1
export DEBUG_SERVER_URL=http://127.0.0.1:17777/event
export PYTHONUNBUFFERED=1
.venv-gpu/bin/python scripts/logged_run.py .venv-gpu/bin/python scripts/train_rhd_supervised.py --config experiments/rhd_supervised_v2.json --run-id rhd_debug_pre_v2
.venv-gpu/bin/python scripts/logged_run.py .venv-gpu/bin/python scripts/train_rhd_supervised.py --config experiments/rhd_supervised_v3.json --run-id rhd_debug_pre_v3
