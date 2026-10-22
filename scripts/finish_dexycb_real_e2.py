"""Wait for sealed main matrix, then verify and run authorized real sync ablation."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/"artifacts"
RUN = ART/"dex_real_e2_matrix_20261020_02"
STAGE = ART/"dex_real_e2_postprocess_20261020_01"
SYNC = ART/"dex_real_e2_sync_20261020_01"


def execute(*args):
    print(json.dumps({"command": list(map(str, args))}), flush=True)
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
    while not (RUN/"sha256.json").exists():
        if (RUN/"failure.txt").exists():
            raise RuntimeError("main run failed; preserve evidence and inspect")
        print(json.dumps({"waiting_for": "sealed main matrix",
                          "completed_sample_directories": len(list(RUN.glob("sample_*")))}), flush=True)
        time.sleep(30)
    assert json.loads((RUN/"status.json").read_text())["real_e2_executed"]
    hashes = json.loads((RUN/"sha256.json").read_text())
    for name, expected in hashes.items():
        assert hashlib.sha256((RUN/name).read_bytes()).hexdigest() == expected, name
    print(json.dumps({"main_hashes_verified": len(hashes)}), flush=True)
    execute("scripts/report_dexycb_real_e2.py", "--run", RUN,
            "--output", ART/"dex_real_e2_report_20261020_01")
    execute("scripts/verify_real_e2_states.py", "--run", RUN,
            "--output", STAGE/"main_state_verification.json")
    # Only after main is sealed, install the evaluator supporting offset-specific
    # pseudo targets. Main's archived source and metrics remain unchanged.
    shutil.copy2(STAGE/"evaluate_real_e2.py", ROOT/"scripts/evaluate_real_e2.py")
    execute("scripts/run_dexycb_sync_sensitivity.py", "--main-run", RUN,
            "--preparation", ART/"dex_real_e2_prepare_20261020_02",
            "--data", ROOT/"inputs/dexycb_official_subset_20261020_01",
            "--run-id", SYNC.name)
    execute("scripts/report_dexycb_sync_sensitivity.py", "--main-run", RUN,
            "--sync-run", SYNC, "--output", ART/"dex_real_e2_sync_report_20261020_01")
    execute("scripts/verify_real_e2_states.py", "--run", SYNC,
            "--output", STAGE/"sync_state_verification.json")
    (STAGE/"completed.json").write_text(json.dumps({"completed": True, "E3_started": False})+"\n")


if __name__ == "__main__":
    main()
