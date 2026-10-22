#!/usr/bin/env bash
set -euo pipefail
cd /workspace/pose_reconstract
config=configs/resnet50_real_mixed_m_s1_sampler_correction.json
run_id=resnet50_real_mixed_m_s1_sampler_correction_20261021_01
mkdir -p logs
exec > >(tee -a "logs/${run_id}_pipeline.log") 2>&1
test ! -e "artifacts/${run_id}"
.venv-gpu/bin/python - <<'PY'
import json
from pathlib import Path
source = Path("artifacts/resnet50_real_mixed_m_s1_subject02_20261021_01")
report = json.loads((source / "results.json").read_text())
verification = json.loads((source / "verification.json").read_text())
assert verification["passed"]
assert report["completion"]["stop_reason"] in ("budget_completed", "dev_patience")
assert report["best"] is not None
assert report["best"]["domains"]["dexycb"]["wrist_relative_mpjpe_mm"] < (
    report["baseline"]["dexycb"]["wrist_relative_mpjpe_mm"])
assert report["domain_imbalance"], "No severe final imbalance; correction is not triggered"
print(json.dumps({"sampler_correction_triggered": True,
                  "source_best_step": report["completion"]["best_step"],
                  "source_domain_imbalance": report["domain_imbalance"]}), flush=True)
PY
.venv-gpu/bin/python scripts/train_resnet50_m_s1.py --config "$config" --freeze
.venv-gpu/bin/python - <<'PY'
import json
from pathlib import Path
old = json.loads(Path("artifacts/m_s1_frozen_splits_20261021_01/freeze.json").read_text())
new = json.loads(Path("artifacts/m_s1_frozen_splits_sampler_correction_20261021_01/freeze.json").read_text())
assert old["split_files"] == new["split_files"]
assert old["subjects"] == new["subjects"]
assert old["sequences"] == new["sequences"]
for path, digest in old["data_hashes"].items():
    if path.endswith("freihand_indices.json"):
        matches = [v for k, v in new["data_hashes"].items() if k.endswith("freihand_indices.json")]
        assert matches == [digest]
    else:
        assert new["data_hashes"][path] == digest
print(json.dumps({"all_three_split_hashes_unchanged": True}), flush=True)
PY
.venv-gpu/bin/python scripts/audit_m_s1_data.py --config "$config"
.venv-gpu/bin/python scripts/train_resnet50_m_s1.py --config "$config" --run-id "$run_id" --device cuda:1
.venv-gpu/bin/python scripts/report_m_s1.py --run "artifacts/${run_id}"
.venv-gpu/bin/python scripts/verify_m_s1_artifacts.py --run "artifacts/${run_id}"
