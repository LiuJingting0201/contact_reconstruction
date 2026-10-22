"""Compare strict calibrated adapter to the preserved upstream fixture."""
import hashlib
import json
from pathlib import Path
import numpy as np
from calibrated_camera import load_calibrated_camera

root = Path(__file__).resolve().parents[1]
out = root/"artifacts/calibrated_camera_v1"
out.mkdir(exist_ok=False)
source = root/"artifacts/camera_loader_probe/cameras.npz"
with np.load(source) as data:
    original = {k: data[k] for k in data.files}
rows = []
for seed in [42, 43]:
    np.random.seed(seed)
    loaded = load_calibrated_camera(source, (1920, 1080), expected_frames=2)
    np.testing.assert_array_equal(loaded["intrins"], original["intrins"])
    np.testing.assert_array_equal(loaded["cam_R"], original["w2c"][:, :3, :3])
    np.testing.assert_array_equal(loaded["cam_t"], original["w2c"][:, :3, 3])
    rows.append({"case": f"preserves_K_R_t_seed_{seed}", "passed": True,
                 "max_translation_error_m": float(np.max(abs(loaded["cam_t"]-original["w2c"][:, :3, 3])))})

# Independent rig point checks baseline sign and world-to-camera convention.
stereo_data = dict(original)
stereo_data["w2c"] = np.tile(np.eye(4), (2, 1, 1))
stereo_data["w2c"][:, 0, 3] = -.06
right_path = out/"right.npz"
np.savez(right_path, **stereo_data)
right = load_calibrated_camera(right_path, (1920, 1080), 2)
p = np.array([.02, .01, .6])
pc = right["cam_R"][0]@p + right["cam_t"][0]
fx = right["intrins"][0, 0]
disparity = fx*p[0]/p[2] - fx*pc[0]/pc[2]
assert abs(disparity-100.) < 1e-10
rows.append({"case": "stereo_baseline_disparity", "passed": True, "disparity_px": float(disparity)})

cases = []
bad = dict(original); bad["intrins"] = original["intrins"].copy(); bad["intrins"][0, 0] = np.nan
cases.append(("nan", bad, (1920,1080), 2))
bad = dict(original); bad["w2c"] = original["w2c"].copy(); bad["w2c"][0, 0, 0] = -1
cases.append(("reflection", bad, (1920,1080), 2))
bad = dict(original); del bad["intrins"]
cases.append(("missing_K", bad, (1920,1080), 2))
cases.append(("wrong_resolution", original, (1280,720), 2))
cases.append(("wrong_frames", original, (1920,1080), 3))
for name, values, size, n in cases:
    path = out/(name+".npz")
    np.savez(path, **values)
    try:
        load_calibrated_camera(path, size, n)
        raise AssertionError("invalid calibration accepted")
    except ValueError as exc:
        rows.append({"case": name, "passed": True, "rejection": str(exc)})
(out/"results.json").write_text(json.dumps(rows, indent=2)+"\n")
(out/"sha256.json").write_text(json.dumps({
    str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in [source, Path(__file__), root/"scripts/calibrated_camera.py", *out.glob("*.npz"), out/"results.json"]
}, indent=2)+"\n")
print(json.dumps(rows, indent=2))
