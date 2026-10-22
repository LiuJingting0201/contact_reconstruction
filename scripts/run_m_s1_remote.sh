#!/usr/bin/env bash
set -euo pipefail
cd /workspace/pose_reconstract
config=configs/resnet50_real_mixed_m_s1_subject02.json
run_id=resnet50_real_mixed_m_s1_subject02_20261021_01
mkdir -p logs
exec > >(tee -a "logs/${run_id}_pipeline.log") 2>&1
test ! -e "artifacts/${run_id}"
.venv-gpu/bin/python scripts/train_resnet50_m_s1.py --config "$config" --freeze
.venv-gpu/bin/python scripts/audit_m_s1_data.py --config "$config"
.venv-gpu/bin/python scripts/train_resnet50_m_s1.py --config "$config" --run-id "$run_id" --device cuda:1
.venv-gpu/bin/python scripts/report_m_s1.py --run "artifacts/${run_id}"
.venv-gpu/bin/python scripts/verify_m_s1_artifacts.py --run "artifacts/${run_id}"
