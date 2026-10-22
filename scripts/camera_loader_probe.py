"""Execute exact upstream camera definitions in isolation; no MANO dependency."""
import ast
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch

root = Path("/workspace/pose_reconstract")
source = root / "vendor/pinned/Dyn-HaMR/dyn-hamr/data/dataset.py"
out = root / "artifacts/camera_loader_probe"
out.mkdir(exist_ok=False)
text = source.read_text()
tree = ast.parse(text)
nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
         and n.name in {"CameraData", "load_cameras_npz"}]
assert len(nodes) == 2

class Logger:
    @staticmethod
    def log(message):
        print(message)

scope = {"np": np, "torch": torch, "os": os, "Logger": Logger}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
camera_path = out / "cameras.npz"
w2c = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
w2c[:, :3, 3] = [.03, -.02, .01]
intrins = np.tile([1000., 930., 931., 517.], (2, 1)).astype(np.float32)
np.savez(camera_path, height=1080, width=1920, focal=1000., w2c=w2c, intrins=intrins)
records = []
for is_static, seed in [(True, 42), (False, 42), (False, 43)]:
    torch.manual_seed(seed)
    camera = scope["CameraData"](str(out), 2, (1920, 1080), is_static)
    actual_K = camera.intrins.numpy()
    actual_t = camera.cam_t.numpy()
    records.append({"is_static": is_static, "seed": seed,
                    "input_intrins": intrins.tolist(), "output_intrins": actual_K.tolist(),
                    "input_translation": w2c[:, :3, 3].tolist(),
                    "output_translation": actual_t.tolist(),
                    "K_max_abs_error_px": float(np.max(np.abs(actual_K-intrins))),
                    "translation_max_error_m": float(np.linalg.norm(actual_t-w2c[:, :3, 3], axis=1).max())})
record = {"test": "isolated exact upstream CameraData, not full pipeline",
          "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "cases": records}
(out/"results.json").write_text(json.dumps(record, indent=2)+"\n")
(out/"executed_source.py").write_text(ast.unparse(ast.Module(body=nodes, type_ignores=[]))+"\n")
print(json.dumps(record, indent=2))
assert records[0]["K_max_abs_error_px"] > 0
assert records[1]["K_max_abs_error_px"] == 0
assert records[1]["output_translation"] != records[2]["output_translation"]
