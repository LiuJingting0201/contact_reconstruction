"""Independent NumPy evaluation of persisted real-E2 states, never optimizes."""
import argparse
import json
from pathlib import Path
import numpy as np


def summary(values):
    a = np.asarray(values, float)
    finite = a[np.isfinite(a)]
    return {"mean": float(finite.mean()) if len(finite) else None,
            "median": float(np.median(finite)) if len(finite) else None,
            "p95": float(np.percentile(finite, 95)) if len(finite) else None,
            "finite_count": int(len(finite)), "total_count": int(len(a)),
            "nonfinite_count": int((~np.isfinite(a)).sum())}


def evaluate(state, target, runtime, config):
    """Recompute projections and GT metrics; no import from runner/adapter."""
    xyz, K, T = state["xyz"], target["K"], target["T"]
    cam = np.einsum("vab,jb->vja", T[:, :3, :3], xyz) + T[:, None, :3, 3]
    h = np.einsum("vab,vjb->vja", K, cam)
    uv = h[..., :2]/np.maximum(h[..., 2:], 1e-6)
    if np.isfinite(uv).all():
        np.testing.assert_allclose(uv, state["uv"], atol=.003, rtol=1e-5)
    gt, valid = target["gt"], target["valid"].astype(bool)
    error2d = np.linalg.norm(uv-target["gt_uv"], axis=-1)
    per_view = np.array([error2d[v][valid[v]].mean() for v in range(2)])
    crop_scale = np.abs(target["affine"][:, 0, 0])
    gt_pseudo = np.linalg.norm(target["pseudo"]-target["gt_uv"], axis=-1)
    rdelta = state["rotations"] @ target["rotations"].swapaxes(-1, -2)
    angles = np.degrees(np.arccos(np.clip((np.trace(rdelta, axis1=-2, axis2=-1)-1)/2, -1, 1)))[0]
    same = np.all(state["rotations"] == target["rotations"], axis=(-2, -1))[0]
    angles[same] = 0
    finite = np.isfinite(xyz).all() and np.isfinite(state["objective"]).all()
    depth_bad = cam[..., 2] <= config["minimum_depth_m"]
    values = {
        "left_gt2d_px": per_view[0], "right_gt2d_px": per_view[1],
        "stereo_gt2d_px": per_view.mean(),
        "gt2d_crop_px": (per_view*crop_scale).mean(),
        "pseudo_residual_px": np.linalg.norm(uv-target["pseudo"], axis=-1).mean(),
        "left_pseudo_gt2d_px": gt_pseudo[0][valid[0]].mean(),
        "right_pseudo_gt2d_px": gt_pseudo[1][valid[1]].mean(),
        "root_relative_mpjpe_mm": np.linalg.norm((xyz-xyz[:1])-(gt-gt[:1]), axis=-1).mean()*1000,
        "absolute_mpjpe_mm": np.linalg.norm(xyz-gt, axis=-1).mean()*1000,
        "wrist_position_error_mm": np.linalg.norm(xyz[0]-gt[0])*1000,
        "wrist_depth_error_mm": abs(xyz[0, 2]-gt[0, 2])*1000,
        "positive_depth_violation_fraction": depth_bad.mean(),
        "global_orientation_change_deg": angles[0],
        "articulated_rotation_change_deg": angles[1:].mean(),
        "beta_l2_change": np.linalg.norm(state["betas"]-target["betas"]),
        "translation_change_mm": np.linalg.norm(state["translation"]-target["translation"])*1000,
        "backend_s": runtime["backend_s"],
        "optimizer_steps": runtime["optimizer_steps"],
        "closure_calls": runtime["closure_calls"],
        "forward_objective_calls": runtime["forward_objective_calls"],
        "peak_allocated_gpu_bytes": runtime["peak_allocated_gpu_bytes"],
    }
    if "gt_camera" in target:
        delivered = target["gt_camera"]
        for v, name in enumerate(("left", "right")):
            values[f"{name}_absolute_mpjpe_mm"] = np.linalg.norm(cam[v]-delivered[v], axis=-1).mean()*1000
            values[f"{name}_wrist_depth_error_mm"] = abs(cam[v, 0, 2]-delivered[v, 0, 2])*1000
            values[f"{name}_wrist_position_error_mm"] = np.linalg.norm(cam[v, 0]-delivered[v, 0])*1000
        values["shared_frame_right_gt_mpjpe_mm"] = np.linalg.norm(
            xyz-target["right_gt_in_left"], axis=-1).mean()*1000
    numerical = not finite or bool(runtime["numerical_failure"])
    quality = (not finite or values["gt2d_crop_px"] > config["quality_gt2d_crop_px"]
               or values["root_relative_mpjpe_mm"] > config["quality_relative_mpjpe_mm"]
               or values["wrist_depth_error_mm"] > config["quality_wrist_depth_mm"])
    values.update(numerical_failure=float(numerical), quality_failure=float(quality),
                  positive_depth_failure=float(depth_bad.any()),
                  objective_increased=float(state["objective"] > state["initial_objective"]+1e-4))
    result = {k: float(v) if np.isfinite(v) else None for k, v in values.items()}
    result["each_articulated_rotation_change_deg"] = [float(x) if np.isfinite(x) else None for x in angles[1:]]
    result["beta_change"] = (state["betas"]-target["betas"]).tolist()
    result["translation_change_m"] = (state["translation"]-target["translation"]).tolist()
    return result


def evaluate_run(run):
    run = Path(run)
    config = json.loads((run/"config.json").read_text())
    rows = []
    for folder in sorted(run.glob("sample_*")):
        sample = json.loads((folder/"sample.json").read_text())
        target = np.load(folder/"evaluation.npz")
        for variant in config["variants"]:
            state_path = folder/variant["id"]/"state.npz"
            if not state_path.exists():
                continue
            runtime = json.loads((state_path.parent/"runtime.json").read_text())
            specific = state_path.parent/"evaluation.npz"
            variant_target = np.load(specific) if specific.exists() else target
            metrics = evaluate(np.load(state_path), variant_target, runtime, config)
            rows.append({"sample_id": sample["sample_id"], "sequence_id": sample["sequence_id"],
                         "variant": variant["id"], "metrics": metrics})
    aggregates = {}
    for variant in config["variants"]:
        selected = [r["metrics"] for r in rows if r["variant"] == variant["id"]]
        aggregates[variant["id"]] = {
            "samples": len(selected),
            "metrics": {k: summary([r[k] if r[k] is not None else np.nan for r in selected])
                        for k in selected[0] if not isinstance(selected[0][k], list)} if selected else {}
        }
    # Paired differences, retaining sample identity. Descriptive only with few
    # sequences; no IID-frame significance claim for correlated video samples.
    index = {(r["sample_id"], r["variant"]): r for r in rows}
    comparisons = []
    for baseline in ("E2-1", "E2-2"):
        for variant in ("E2-3", "E2-4", "E2-5", "E2-6", "E2-7", "E2-8"):
            for metric in ("absolute_mpjpe_mm", "wrist_position_error_mm", "wrist_depth_error_mm"):
                diffs = []
                for (sid, v), row in index.items():
                    if v != baseline or (sid, variant) not in index:
                        continue
                    a, b = row["metrics"][metric], index[(sid, variant)]["metrics"][metric]
                    if a is not None and b is not None:
                        diffs.append(b-a)
                comparisons.append({"baseline": baseline, "variant": variant, "metric": metric,
                                    "delta_after_minus_before": summary(diffs),
                                    "fraction_improved": float(np.mean(np.array(diffs) < 0)) if diffs else None})
    report = {"rows": rows, "aggregates": aggregates, "paired_comparisons": comparisons,
              "E3_gate": "not automatically opened; inspect paired majority and failures under nonoracle confidence",
              "statistics": "mean/median/p95 over finite samples, with explicit denominator and failure rates",
              "limitations": "No sequence-level uncertainty estimate or sync-offset result without qualified real data."}
    destination = run/"independent_metrics.json"
    if destination.exists():
        raise FileExistsError(f"never overwrite prior evaluation: {destination}")
    destination.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    report = evaluate_run(args.run)
    print(json.dumps({"independently_evaluated_states": len(report["rows"])}))
