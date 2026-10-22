#!/usr/bin/env bash
set -euo pipefail
cd /workspace/pose_reconstract
export CUDA_VISIBLE_DEVICES=1
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export HYDRA_FULL_ERROR=1
.venv-gpu/bin/python - <<'PY'
from pathlib import Path
import hashlib
p = Path("inputs/hamer.ckpt")
h = hashlib.file_digest(p.open("rb"), "sha256").hexdigest() if hasattr(hashlib,"file_digest") else hashlib.sha256(p.read_bytes()).hexdigest()
assert h == "e5cc06f294d88a92dee24e603480aab04de532b49f0e08200804ee7d90e16f53", h
print("Remote checkpoint verified:", h)
PY
.venv-gpu/bin/python scripts/logged_run.py .venv-gpu/bin/python scripts/real_image_inference.py
.venv-gpu/bin/python scripts/logged_run.py .venv-gpu/bin/python scripts/train_resnet_smoke.py
