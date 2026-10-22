"""Independently recompute E1/E2-control metrics from archived raw arrays."""
import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=Path)
args = parser.parse_args()
path = args.directory
report = json.loads((path / "results.json").read_text())
inputs = np.load(path / "inputs.npz")
checks = []
for name, recorded in report["results"].items():
    result = np.load(path / ("initial.npz" if name == "initial" else f"{name}_final.npz"))
    xyz = result["xyz"]
    synthetic = report["experiment"] != "E1"
    views = [0] if name == "left" else [1] if name == "right" else list(range(inputs["K"].shape[1]))
    K, R, ext_t = inputs["K"][:, views], inputs["R"][views], inputs["ext_t"][views]
    camera = np.einsum("vab,njb->nvja", R, xyz) + ext_t[None, :, None]
    homogeneous = np.einsum("nvab,nvjb->nvja", K, camera)
    uv = homogeneous[..., :2] / homogeneous[..., 2:]
    assert np.allclose(uv, result["uv"], atol=2e-4, rtol=1e-6), (name, "projection")
    gt = inputs["gt"]
    gt_uv = inputs["clean"][:, views] if synthetic else inputs["gt_uv"]
    visible = np.ones(gt_uv.shape[:-1]) if synthetic else inputs["visible"]
    scale = np.ones(gt_uv.shape[:2]) if synthetic else inputs["affine_scale"]
    per_view = (np.linalg.norm(uv-gt_uv, axis=-1)*visible).sum(-1)/visible.sum(-1)
    recalculated = {
        "gt2d_full_px": per_view.mean(1), "gt2d_crop_px": (per_view*scale).mean(1),
        "absolute_mpjpe_mm": np.linalg.norm(xyz-gt, axis=-1).mean(1)*1000,
        "root_relative_mpjpe_mm": np.linalg.norm(
            (xyz-xyz[:, :1])-(gt-gt[:, :1]), axis=-1).mean(1)*1000,
        "wrist_depth_error_mm": np.abs(xyz[:, 0, 2]-gt[:, 0, 2])*1000,
        "beta_l2_change": np.linalg.norm(result["betas"]-inputs["betas"], axis=1),
        "translation_change_mm": np.linalg.norm(result["translation"]-inputs["translation"], axis=1)*1000,
    }
    for key, values in recalculated.items():
        assert np.allclose(values, recorded["per_image"][key], atol=2e-4, rtol=1e-5), (name, key)
        assert np.isclose(values.mean(), recorded["mean"][key], atol=2e-4, rtol=1e-5), (name, key)
    numerical = (~np.isfinite(xyz).all((1, 2)) | (camera[..., 2].min((1, 2)) <= .01)
                 | (result["objective"] > result["initial_objective"]+1e-4))
    quality = ((recalculated["gt2d_crop_px"] > 20) |
               (recalculated["root_relative_mpjpe_mm"] > 50) |
               (recalculated["wrist_depth_error_mm"] > 100))
    assert numerical.tolist() == recorded["numerical_failure"]
    assert quality.tolist() == recorded["quality_failure"]
    if name != "initial":
        trace = np.load(path / f"{name}_trajectory.npz")
        assert len(trace["objective"]) == 801
        assert np.allclose(trace["rotations"][0], inputs["rotations"], atol=1e-6)
        assert np.array_equal(trace["translation"][0], inputs["translation"])
        for key in ("rotations", "betas", "translation", "objective"):
            assert np.isfinite(trace[key]).all()
            assert np.array_equal(trace[key][-1], result[key]), (name, key, "final_trace")
        logs = [json.loads(line) for line in (path / f"{name}_steps.jsonl").read_text().splitlines()]
        assert [r["step"] for r in logs] == list(range(801))
        assert np.allclose([r["objective"] for r in logs], trace["objective"])
        if name == "translation":
            assert np.array_equal(result["rotations"], inputs["rotations"])
        if name == "translation_orientation":
            assert np.array_equal(result["rotations"][:, 1:], inputs["rotations"][:, 1:])
        if name != "translation_pose_shape":
            assert np.array_equal(result["betas"], inputs["betas"])
    checks.append(name)
output = {"passed": True, "variants_verified": checks,
          "checks": ["independent projection and metrics", "failure flags", "801-step traces",
                     "identical initialization", "frozen parameter invariants", "finite trajectories"]}
(path / "verification.json").write_text(json.dumps(output, indent=2) + "\n")
print(json.dumps(output, indent=2))
