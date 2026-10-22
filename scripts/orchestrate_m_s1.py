"""Finish local input preparation and run the authorized remote M-S1 pipeline."""
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "logs/m_s1_orchestration_state.json"
REMOTE = "simready-dev:/workspace/pose_reconstract/"


def status(stage, **kwargs):
    value = {"stage": stage, "updated_unix": time.time(), **kwargs}
    STATE.write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(value), flush=True)


def run(args):
    print(json.dumps({"command": args}), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def main():
    archive = ROOT / "inputs/dexycb_official_download_20261020/20200928-subject-07.tar.gz.part"
    status("waiting_subject07_download")
    deadline = time.monotonic() + 3 * 3600
    while not archive.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("subject07 download did not finish")
        time.sleep(10)
    status("auditing_extracting_subject07")
    run([".venv-dex-download/bin/python", "scripts/prepare_dexycb_full_m_s1.py",
         "--archive", str(archive),
         "--calibration", "inputs/dexycb_official_download_20261020/calibration_gdown.tar.gz",
         "--audit", "artifacts/dexycb_subject07_archive_audit_20261021_01",
         "--output", "inputs/dexycb_m_s1/20200928-subject-07", "--split", "dev"])
    status("waiting_subject02_manifest")
    train_manifest = ROOT / "inputs/dexycb_m_s1/20200813-subject-02/sealed_manifest.json"
    deadline = time.monotonic() + 3600
    while not train_manifest.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("subject02 preparation did not finish")
        time.sleep(10)
    status("syncing_all_domains")
    run(["rsync", "-a", "inputs/dexycb_m_s1/", REMOTE + "inputs/dexycb_m_s1/"])
    # Recheck full FreiHAND after the initial background transfer.
    run(["rsync", "-a", "inputs/freihand_v2/training/rgb/",
         REMOTE + "inputs/freihand_v2/training/rgb/"])
    run(["rsync", "-a",
         "scripts/train_resnet50_m_s1.py", "scripts/train_resnet50_real_mixed.py",
         "scripts/audit_m_s1_data.py", "scripts/report_m_s1.py", "scripts/run_m_s1_remote.sh",
         "scripts/verify_m_s1_artifacts.py",
         REMOTE + "scripts/"])
    run(["rsync", "-a", "configs/resnet50_real_mixed_m_s1_subject02.json", REMOTE + "configs/"])
    status("remote_freeze_audit_and_train")
    run(["ssh", "simready-dev",
         "bash /workspace/pose_reconstract/scripts/run_m_s1_remote.sh"])
    status("syncing_results")
    run(["rsync", "-a", REMOTE + "artifacts/resnet50_real_mixed_m_s1_subject02_20261021_01/",
         "artifacts/resnet50_real_mixed_m_s1_subject02_20261021_01/"])
    run(["rsync", "-a", REMOTE + "artifacts/m_s1_frozen_splits_20261021_01/",
         "artifacts/m_s1_frozen_splits_20261021_01/"])
    run(["rsync", "-a", REMOTE + "artifacts/m_s1_coordinate_audit_20261021_01/",
         "artifacts/m_s1_coordinate_audit_20261021_01/"])
    status("complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        status("failed", error=repr(error))
        raise
