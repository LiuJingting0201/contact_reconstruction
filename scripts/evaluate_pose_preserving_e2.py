"""Independent NumPy metrics for the cached real-E2 pose-preserving ablation."""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FINGERS = {"thumb": [0, 1, 2, 3, 4], "index": [0, 5, 6, 7, 8],
           "middle": [0, 9, 10, 11, 12], "ring": [0, 13, 14, 15, 16],
           "little": [0, 17, 18, 19, 20]}
TIPS = [4, 8, 12, 16, 20]


def stats(values):
    x = np.asarray(values, float)
    assert np.isfinite(x).all()
    return {"mean": float(x.mean()), "median": float(np.median(x)),
            "p95": float(np.percentile(x, 95)), "count": int(len(x))}


def axis_angle_matrix(vector):
    theta = np.linalg.norm(vector)
    if theta < 1e-12:
        return np.eye(3)
    x, y, z = vector/theta
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0.]])
    return np.eye(3)+np.sin(theta)*K+(1-np.cos(theta))*(K@K)


def rotation_angle(a, b):
    q = a @ b.T
    return float(np.degrees(np.arccos(np.clip((np.trace(q)-1)/2, -1, 1))))


def aligned_error(pred, gt):
    x, y = pred[1:]-pred[:1], gt[1:]-gt[:1]
    u, _, vt = np.linalg.svd(x.T@y)
    R = vt.T@u.T
    if np.linalg.det(R) < 0:
        vt[-1] *= -1
        R = vt.T@u.T
    assert np.linalg.det(R) > 0.999999
    return np.linalg.norm(x@R.T-y, axis=-1).mean()*1000


def bone_errors(pred, gt):
    result = {}
    for name, chain in FINGERS.items():
        p, g = pred[chain], gt[chain]
        p, g = np.diff(p, axis=0), np.diff(g, axis=0)
        cosine = np.sum(p*g, axis=-1)/(np.linalg.norm(p, axis=-1)*np.linalg.norm(g, axis=-1))
        result[name] = float(np.degrees(np.arccos(np.clip(cosine, -1, 1))).mean())
    return result


def one(state, target, runtime, row, config):
    xyz, gt, K, T = state["xyz"], target["gt"], target["K"], target["T"]
    cam = np.einsum("vab,jb->vja", T[:, :3, :3], xyz)+T[:, None, :3, 3]
    h = np.einsum("vab,vjb->vja", K, cam)
    uv = h[..., :2]/h[..., 2:]
    np.testing.assert_allclose(uv, state["uv"], atol=.003, rtol=1e-5)
    valid = target["valid"].astype(bool)
    e2 = np.linalg.norm(uv-target["gt_uv"], axis=-1)
    rel = (xyz-xyz[:1])-(gt-gt[:1])
    tip_abs = np.linalg.norm(xyz[TIPS]-gt[TIPS], axis=-1)*1000
    tip_rel = np.linalg.norm(rel[TIPS], axis=-1)*1000
    delta = state["rotations"]@target["rotations"].swapaxes(-1, -2)
    angles = np.degrees(np.arccos(np.clip((np.trace(delta, axis1=-2, axis2=-1)-1)/2, -1, 1)))[0]
    exact = np.all(state["rotations"] == target["rotations"], axis=(-2, -1))[0]
    angles[exact] = 0
    gt_global = axis_angle_matrix(np.asarray(row["views"][0]["pose_m"])[0, :3])
    metrics = {
        "absolute_mpjpe_mm": np.linalg.norm(xyz-gt, axis=-1).mean()*1000,
        "wrist_relative_mpjpe_mm": np.linalg.norm(rel, axis=-1).mean()*1000,
        "rotation_aligned_wrist_relative_mpjpe_mm_diagnostic": aligned_error(xyz, gt),
        "wrist_position_error_mm": np.linalg.norm(xyz[0]-gt[0])*1000,
        "left_wrist_depth_error_mm": abs(cam[0, 0, 2]-target["gt_camera"][0, 0, 2])*1000,
        "right_wrist_depth_error_mm": abs(cam[1, 0, 2]-target["gt_camera"][1, 0, 2])*1000,
        "fingertip_absolute_error_mm": tip_abs.mean(),
        "fingertip_wrist_relative_error_mm": tip_rel.mean(),
        "left_gt2d_epe_px": e2[0][valid[0]].mean(),
        "right_gt2d_epe_px": e2[1][valid[1]].mean(),
        "pseudo_residual_px": np.linalg.norm(uv-target["pseudo"], axis=-1).mean(),
        "global_orientation_error_deg": rotation_angle(state["rotations"][0, 0], gt_global),
        "translation_change_mm": np.linalg.norm(state["translation"]-target["translation"])*1000,
        "global_rotation_change_deg": angles[0],
        "articulated_pose_change_deg": angles[1:].mean(),
        "shape_change_l2": np.linalg.norm(state["betas"]-target["betas"]),
        "backend_s": runtime["backend_s"],
        "numerical_failure": float(runtime["numerical_failure"] is not None),
        "positive_depth_failure": float((cam[..., 2] <= config["minimum_depth_m"]).any()),
    }
    metrics.update({f"bone_{name}_angular_error_deg": value
                    for name, value in bone_errors(xyz-xyz[:1], gt-gt[:1]).items()})
    crop_scale = np.abs(target["affine"][:, 0, 0])
    crop_epe = np.mean([e2[v][valid[v]].mean()*crop_scale[v] for v in range(2)])
    metrics["quality_failure"] = float(crop_epe > config["quality_gt2d_crop_px"]
        or metrics["wrist_relative_mpjpe_mm"] > config["quality_relative_mpjpe_mm"]
        or metrics["left_wrist_depth_error_mm"] > config["quality_wrist_depth_mm"])
    assert np.isfinite(list(metrics.values())).all()
    return {k: float(v) for k, v in metrics.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    config = json.loads((a.run/"config.json").read_text())
    rows = []
    reference_rows = []
    source = ROOT/config["source_run"]
    for folder in sorted(a.run.glob("sample_*")):
        row = json.loads((folder/"sample.json").read_text())
        target = np.load(folder/"evaluation.npz")
        for variant in config["variants"]:
            path = folder/variant["id"]
            metrics = one(np.load(path/"state.npz"), target,
                          json.loads((path/"runtime.json").read_text()), row, config)
            rows.append({"sample_id": row["sample_id"], "sequence_id": row["sequence_id"],
                         "variant": variant["id"], "metrics": metrics})
        reference = source/folder.name/"E2-2"
        reference_rows.append({"sample_id": row["sample_id"], "sequence_id": row["sequence_id"],
            "variant": "E2-2-reference", "metrics": one(np.load(reference/"state.npz"), target,
            json.loads((reference/"runtime.json").read_text()), row, config)})
    assert len(rows) == 32*11
    aggregates = {}
    for variant in config["variants"]:
        selected = [x["metrics"] for x in rows if x["variant"] == variant["id"]]
        aggregates[variant["id"]] = {key: stats([x[key] for x in selected])
                                      for key in selected[0]}
    index = {(x["sample_id"], x["variant"]): x["metrics"] for x in rows}
    index.update({(x["sample_id"], x["variant"]): x["metrics"] for x in reference_rows})
    comparisons = {}
    for candidate, baseline in (("B2", "B0"), ("B2", "E2-2-reference"), ("B2", "B4"),
                                ("B5", "B2"), ("B6", "B2"), ("B7", "B2"),
                                ("B8", "B2"), ("B10", "B2")):
        key = f"{candidate}_vs_{baseline}"
        comparisons[key] = {}
        for metric in ("absolute_mpjpe_mm", "wrist_relative_mpjpe_mm",
                       "rotation_aligned_wrist_relative_mpjpe_mm_diagnostic",
                       "wrist_position_error_mm", "fingertip_absolute_error_mm",
                       "fingertip_wrist_relative_error_mm", "global_orientation_error_deg"):
            ids = sorted({x["sample_id"] for x in rows})
            delta = np.array([index[(sid, candidate)][metric]-index[(sid, baseline)][metric]
                              for sid in ids])
            comparisons[key][metric] = {"delta_candidate_minus_baseline": stats(delta),
                "candidate_improved_count": int((delta < -1e-4).sum()),
                "improvement_tolerance": "delta < -1e-4 in the metric's displayed unit",
                "per_sample": [{"sample_id": sid,
                    "candidate": index[(sid, candidate)][metric],
                    "baseline": index[(sid, baseline)][metric], "delta": float(value)}
                    for sid, value in zip(ids, delta)]}
    reference_aggregate = {key: stats([x["metrics"][key] for x in reference_rows])
                           for key in reference_rows[0]["metrics"]}
    result = {"rows": rows, "aggregates": aggregates, "reference_rows": reference_rows,
              "reference_aggregates": {"E2-2-reference": reference_aggregate},
              "paired_comparisons": comparisons,
              "global_orientation": {"status": "available after separate MANO pose_m reconstruction audit",
               "definition": "SO(3) geodesic prediction versus DexYCB pose_m global axis-angle in left camera"},
              "rotation_aligned_metric": "diagnostic only; SO(3) Kabsch on 20 non-wrist joints, no scale/reflection/translation",
              "gt_use": "evaluation only; no metrics imported by optimizer"}
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"rows": len(rows), "output": str(a.output)}), flush=True)


if __name__ == "__main__":
    main()
