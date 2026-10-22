#!/usr/bin/env bash
set -euo pipefail
cd /workspace/pose_reconstract
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
.venv-gpu/bin/python scripts/logged_run.py .venv-gpu/bin/python scripts/train_rhd_supervised.py --run-id rhd_supervised_v2_run02
.venv-gpu/bin/python scripts/logged_run.py .venv-gpu/bin/python scripts/summarize_rhd.py --run-id rhd_supervised_v2_run02
# Finalize hashes separately after this launcher's redirected log is closed.
