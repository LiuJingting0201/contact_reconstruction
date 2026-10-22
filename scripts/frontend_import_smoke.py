"""Load upstream HaMeR Python interfaces without inventing model assets."""
import hashlib
import importlib
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

root = Path("/workspace/pose_reconstract")
sys.path.insert(0, str(root/"repos/hamer"))
results = []
for module in ["torch", "torchvision", "pytorch_lightning", "smplx", "timm",
               "cv2", "pyrender", "hamer.models"]:
    try:
        loaded = importlib.import_module(module)
        results.append({"module": module, "status": "ok", "path": loaded.__file__,
                        "version": getattr(loaded, "__version__", None)})
    except Exception:
        results.append({"module": module, "status": "failed", "traceback": traceback.format_exc()})
assets = {
    "checkpoint": root/"repos/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt",
    "MANO_RIGHT": root/"repos/hamer/_DATA/data/mano/MANO_RIGHT.pkl",
}
record = {"kind": "imports_only_not_model_inference", "modules": results,
          "assets": {n: {"path": str(p), "exists": p.is_file()} for n,p in assets.items()},
          "hamer_source_sha256": hashlib.sha256((root/"repos/hamer/hamer/models/hamer.py").read_bytes()).hexdigest()}
out = root/"artifacts/bootstrap"/("frontend_import_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+".json")
out.write_text(json.dumps(record, indent=2)+"\n")
print(json.dumps(record, indent=2))
sys.exit(int(any(r["status"] != "ok" for r in results)))
