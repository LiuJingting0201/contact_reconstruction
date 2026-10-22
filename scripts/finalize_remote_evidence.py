"""Snapshot environment and hash remote evidence after experiments."""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root/"artifacts/bootstrap"
proc = subprocess.run([str(root/".venv-gpu/bin/python"), "-m", "pip", "freeze"],
                      text=True, capture_output=True, check=True)
(out/"freeze-gpu-final.txt").write_text(proc.stdout)
inventory = {}
for name in ["hamer", "Fast-HaMeR"]:
    base = root/"repos"/name
    inventory[name] = {
        str(p.relative_to(base)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(base.rglob("*")) if p.is_file() and "__pycache__" not in p.parts
    }
(out/"source_inventory.json").write_text(json.dumps(inventory, indent=2)+"\n")
files = {}
for path in sorted((root/"artifacts").rglob("*")):
    if path.is_file() and path.name != "remote_sha256.json" and "commands" not in path.parts:
        files[str(path.relative_to(root/"artifacts"))] = hashlib.sha256(path.read_bytes()).hexdigest()
(root/"artifacts/remote_sha256.json").write_text(json.dumps({
    "timestamp_utc": datetime.now(timezone.utc).isoformat(), "files": files,
    "excluded": "commands (active wrapper logs); each completed command has its own hashes"
}, indent=2)+"\n")
print(json.dumps({"hashed_evidence_files": len(files), "source_files": {k:len(v) for k,v in inventory.items()}}))
