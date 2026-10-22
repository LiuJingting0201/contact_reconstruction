"""Seal scoped real-E2 evidence, or verify an existing remote manifest locally."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", type=Path)
    a = parser.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if a.verify:
        expected = json.loads(a.verify.read_text())["files"]
        missing, mismatch = [], []
        for name, value in expected.items():
            path = ROOT/name
            if not path.is_file():
                missing.append(name)
            elif digest(path) != value:
                mismatch.append(name)
        old = json.loads((ROOT/"artifacts/remote/remote_sha256.json").read_text())["files"]
        changed_old = [name for name, value in old.items()
                       if digest(ROOT/"artifacts/remote"/name) != value]
        result = {"passed": not (missing or mismatch or changed_old),
                  "remote_files_verified": len(expected), "missing": missing,
                  "mismatch": mismatch, "inherited_files_verified": len(old),
                  "inherited_changes": changed_old, "optimizer_updates_performed": 0}
    else:
        paths = []
        for folder in (ROOT/"artifacts").glob("dex*"):
            if not folder.is_dir() or "_preview_" in folder.name:
                continue
            paths.extend(p for p in folder.rglob("*") if p.is_file())
        paths.extend(p for p in (ROOT/"artifacts").glob("dex*.log") if p.is_file())
        paths.extend(p for p in (ROOT/"inputs/dexycb_official_subset_20261020_01").rglob("*") if p.is_file())
        script_names = ("run_real_e2.py", "real_e2_data.py", "evaluate_real_e2.py",
                        "run_mano_backend.py", "prepare_dexycb_real_e2.py",
                        "verify_dexycb_official_acquisition.py", "report_dexycb_real_e2.py",
                        "verify_real_e2_states.py", "run_dexycb_sync_sensitivity.py",
                        "report_dexycb_sync_sensitivity.py", "finish_dexycb_real_e2.py",
                        "make_dexycb_review_contact.py", "audit_dexycb_camera_branches.py",
                        "run_pose_preserving_e2.py", "evaluate_pose_preserving_e2.py",
                        "verify_pose_preserving_e2.py", "report_pose_preserving_e2.py",
                        "seal_dexycb_delivery.py")
        paths.extend(ROOT/"scripts"/name for name in script_names)
        paths.extend(ROOT/"docs"/name for name in ("dyn_hamr_experiment_record.md",
                     "dyn_hamr_reproduction_protocol.md", "real_e2_input_contract.md",
                     "dexycb_real_e2_data_audit.md"))
        paths.extend((ROOT/"configs/real_e2_dexycb_official_v2.json",
                      ROOT/"configs/pose_preserving_v1.json"))
        files = {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(paths))
                 if p.resolve() != a.output.resolve()}
        result = {"files": files, "count": len(files),
                  "scope": "real E2 artifacts, selected official inputs, current code/config/docs; excludes local preview and full subject archive",
                  "full_archive_hash": "artifacts/dexycb_official_acquisition_20261020_01/verification.json"}
    a.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k != "files"}), flush=True)
    if result.get("passed") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
