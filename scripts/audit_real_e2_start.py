"""Read-only inventory of inherited evidence; write only a new audit directory."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "artifacts/real_e2_audit_20261020_01"
out.mkdir(exist_ok=False)
git = subprocess.run(["git", "status", "--porcelain=v1"], cwd=ROOT, capture_output=True, text=True)
(out / "git_status.json").write_text(json.dumps({"returncode": git.returncode, "stdout": git.stdout,
                                                "stderr": git.stderr, "commit": None}, indent=2))
inventory = {}
for dirname in ("scripts", "docs", "experiments"):
    for p in sorted((ROOT / dirname).rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            dest = out / "before" / p.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
            inventory[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
(out / "source_before_sha256.json").write_text(json.dumps(inventory, indent=2))
base = ROOT / ("artifacts/remote" if (ROOT / "artifacts/remote/e1_mano_run02").exists() else "artifacts")
manifest = json.loads((base / "remote_sha256.json").read_text())
errors = []
for name, expected in manifest["files"].items():
    p = base / name
    actual = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
    if actual != expected:
        errors.append({"file": name, "expected": expected, "actual": actual})
(out / "inherited_hash_check.json").write_text(json.dumps({
    "count": len(manifest["files"]), "errors": errors}, indent=2))
assert not errors, errors
artifacts = [{"path": str(p.relative_to(ROOT)), "size": p.stat().st_size}
             for p in sorted((ROOT / "artifacts").rglob("*")) if p.is_file() and out not in p.parents]
(out / "artifact_inventory.json").write_text(json.dumps(artifacts, indent=2))
for run in ("e1_mano_run02", "e2_synthetic_run02"):
    directory = base / run
    report = json.loads((directory / "results.json").read_text())
    print(run, json.dumps({k: v for k, v in report.items() if k != "results"}))
    diff = "".join(difflib.unified_diff((directory / "run_mano_backend.py").read_text().splitlines(True),
                                       (ROOT / "scripts/run_mano_backend.py").read_text().splitlines(True),
                                       fromfile=f"{run}/snapshot", tofile="current/script"))
    (out / f"{run}_current.diff").write_text(diff)
    for variant, row in report["results"].items():
        print(variant, json.dumps(row, separators=(",", ":")))
    arrays = {}
    for p in directory.glob("*.npz"):
        with np.load(p, allow_pickle=False) as data:
            arrays[p.name] = {k: {"shape": list(data[k].shape), "dtype": str(data[k].dtype),
                                  "finite": bool(np.isfinite(data[k]).all())} for k in data.files}
    (out / f"{run}_array_inventory.json").write_text(json.dumps(arrays, indent=2))
print(json.dumps({"inherited_files_verified": len(manifest["files"]), "artifact_count": len(artifacts)}))
