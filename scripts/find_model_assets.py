"""Search only established remote project/data roots for hand-model resources."""
import json
import os
from pathlib import Path
roots = ["/workspace", "/opt/simready", "/mnt/session"]
hits, errors = [], []
skip = {".git", "node_modules", "site-packages", ".uv-cache", "__pycache__"}
for root in roots:
    for directory, dirs, files in os.walk(root, onerror=lambda e: errors.append(str(e))):
        dirs[:] = [d for d in dirs if d not in skip and not d.startswith(".venv")]
        for name in files:
            lower = name.lower()
            if (lower.startswith("mano") and lower.endswith((".pkl", ".npz"))) or \
               ("hamer" in lower and lower.endswith((".ckpt", ".pth", ".pt", ".yaml"))) or \
               name == "cameras.npz":
                p = Path(directory) / name
                hits.append({"path": str(p), "bytes": p.stat().st_size})
record = {"scope": roots, "hits": hits, "errors": errors}
out = Path("/workspace/pose_reconstract/artifacts/bootstrap/asset_inventory.json")
out.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
