#!/usr/bin/env bash
set -euo pipefail
cd /workspace/pose_reconstract
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
.venv-gpu/bin/python scripts/logged_run.py .venv-gpu/bin/python scripts/train_rhd_supervised.py --config experiments/rhd_supervised_v3.json
.venv-gpu/bin/python scripts/logged_run.py .venv-gpu/bin/python scripts/summarize_rhd.py --run-id rhd_supervised_v3
# Finalize hashes separately after this launcher's redirected log is closed.
