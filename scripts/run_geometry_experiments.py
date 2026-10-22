#!/usr/bin/env python3
"""CPU synthetic geometry experiments. Does not execute HaMeR or MANO."""
import argparse
import csv
import hashlib
import json
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def json_write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project(points, K, min_depth=0.0):
    if not np.isfinite(points).all() or np.any(points[:, 2] <= min_depth):
        raise ValueError("nonfinite_or_nonpositive_depth")
    homogeneous = points @ K.T
    return homogeneous[:, :2] / homogeneous[:, 2:]


def translation_fit(local, uv, K, weights, cfg):
    valid = np.isfinite(uv).all(axis=1) & np.isfinite(local).all(axis=1)
    valid &= np.isfinite(weights) & (weights > 0)
    if valid.sum() < 3:
        raise ValueError("insufficient_observations")
    p, q, w = local[valid], uv[valid], weights[valid]
    xyh = np.c_[q, np.ones(len(q))] @ np.linalg.inv(K).T
    xy = xyh[:, :2] / xyh[:, 2:]
    A = np.zeros((len(p), 2, 3))
    A[:, 0, 0], A[:, 1, 1] = -1, -1
    A[:, :, 2] = xy
    b = p[:, :2] - xy * p[:, 2:]
    A = (A * np.sqrt(w[:, None, None])).reshape(-1, 3)
    b = (b * np.sqrt(w[:, None])).reshape(-1)
    t, _, rank, singular = np.linalg.lstsq(A, b, rcond=None)
    cond = singular[0] / singular[-1] if singular[-1] > 0 else np.inf
    if rank < 3 or cond > cfg["max_condition"]:
        raise ValueError("degenerate_geometry")
    if np.any((local[valid] + t)[:, 2] <= cfg["min_depth_m"]):
        raise ValueError("nonpositive_depth_solution")
    return t, float(cond), valid


def fit_shared_translation(local, obs, cameras, weights, init, cfg, robust, trace):
    # Each camera is (K, world-to-camera R, t); local is already world-oriented.
    active = []
    for v, (K, R, t) in enumerate(cameras):
        valid = np.isfinite(obs[v]).all(axis=1) & np.isfinite(weights[v]) & (weights[v] > 0)
        if valid.any():
            active.append((v, K, R, t, valid, weights[v, valid]))
    if not active:
        raise ValueError("no_active_views")

    def residual(trans):
        result = []
        for v, K, R, t, valid, w in active:
            uv = project((local[valid] + trans) @ R.T + t, K)
            r = (uv - obs[v, valid]) / cfg["sigma_px"]
            # scipy minimizes 0.5 * sum(z^2); transform so that this equals
            # mean_view(sum_j w_j sum_xy Huber(r) / sum_j w_j).
            if robust:
                ar = np.abs(r)
                r = np.sign(r) * np.sqrt(np.where(ar <= 1, r*r, 2*ar-1))
            result.append((r * np.sqrt(w[:, None] / (w.sum() * len(active)))).ravel())
        values = np.concatenate(result)
        trace.append({"translation_m": trans.tolist(), "objective": float(.5 * values @ values)})
        return values

    # Tested rig has identity camera rotations; guard prevents false depth bounds.
    if any(not np.allclose(R, np.eye(3)) for _, R, _ in cameras):
        raise ValueError("translation_solver_depth_bound_requires_parallel_cameras")
    lower_z = max(cfg["min_depth_m"] - local[:, 2].min() - t[2] for _, _, t in cameras)
    sol = least_squares(residual, init, bounds=([-np.inf, -np.inf, lower_z], [np.inf]*3),
                        max_nfev=cfg["max_nfev"], ftol=cfg["solver_tolerance"],
                        xtol=cfg["solver_tolerance"], gtol=cfg["solver_tolerance"])
    return sol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/geometry_v1.json")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not args.run_id.replace("-", "").replace("_", "").isalnum():
        raise SystemExit("run-id must be alphanumeric with - or _")
    out = ROOT / "artifacts/geometry" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    cfg_path = ROOT / args.config
    cfg = json.loads(cfg_path.read_text())
    shutil.copy2(cfg_path, out / "config.json")
    shutil.copy2(__file__, out / "executed_script.py")
    shutil.copy2(ROOT / "docs/dyn_hamr_reproduction_protocol.md", out / "protocol_at_run.md")
    start = time.perf_counter()
    manifest = {"run_id": args.run_id, "data_kind": "synthetic_non_MANO",
                "start_utc": datetime.now(timezone.utc).isoformat(), "argv": sys.argv,
                "python": sys.version, "platform": platform.platform(),
                "numpy": np.__version__, "scipy": scipy.__version__,
                "config_sha256": sha(cfg_path), "script_sha256": sha(Path(__file__))}
    json_write(out / "manifest.json", manifest)
    rng = np.random.default_rng(cfg["seed"])
    # Random non-coplanar point cloud, metric extent specified explicitly here.
    local = rng.uniform([-.04, -.07, -.025], [.04, .07, .025], (cfg["synthetic_points"], 3))
    true_t = np.array(cfg["true_translation_m"])
    Kg, K = np.array(cfg["K_general"]), np.array(cfg["K_rectified"])
    base = cfg["stereo_baseline_m"]
    cameras = [(K, np.eye(3), np.zeros(3)), (K, np.eye(3), np.array([-base, 0, 0]))]
    uv = project(local + true_t, Kg)
    stereo = np.array([project(local + true_t + t, kv) for kv, _, t in cameras])
    raw = {"local_points_m": local, "true_translation_m": true_t,
           "general_uv_px": uv, "stereo_uv_px": stereo}
    results, traces = [], {}

    def record(name, metrics, passed=None, classification="measurement"):
        entry = {"case": name, "classification": classification,
                 "passed": passed, "metrics": metrics}
        results.append(entry)
        with (out / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(entry, allow_nan=False) + "\n")
        print(json.dumps(entry, allow_nan=False), flush=True)

    thresholds = cfg["thresholds"]
    t, cond, _ = translation_fit(local, uv, Kg, np.ones(len(local)), cfg)
    error = float(np.linalg.norm(t-true_t))
    record("general_K_translation", {"error_m": error, "condition": cond},
           error < thresholds["noiseless_translation_m"], "invariant")
    homogeneous = (local+true_t) @ Kg.T
    independent_u = (Kg[0, 0]*(local[:, 0]+true_t[0]) +
                     Kg[0, 1]*(local[:, 1]+true_t[1])) / (local[:, 2]+true_t[2]) + Kg[0, 2]
    projection_error = float(np.max(np.abs(independent_u-homogeneous[:, 0]/homogeneous[:, 2])))
    record("projection_equation", {"max_error_px": projection_error},
           projection_error < thresholds["roundtrip_px"], "invariant")

    # Pixel-center convention mirror x -> crop_size-1-x; matrix includes mirror.
    scale = cfg["crop_size_px"]/cfg["bbox_size_px"]
    cx, cy = cfg["bbox_center_px"]
    side = cfg["crop_size_px"]
    A = np.array([[scale, 0, side/2-scale*cx], [0, scale, side/2-scale*cy], [0, 0, 1.]])
    mirror = np.array([[-1., 0, side-1], [0, 1, 0], [0, 0, 1]])
    image_h = np.c_[uv, np.ones(len(uv))]
    for handedness, transform in [("right", A), ("left", mirror @ A)]:
        crop = image_h @ transform.T
        recovered = crop @ np.linalg.inv(transform).T
        error = float(np.max(np.linalg.norm(recovered[:, :2]-uv, axis=1)))
        record("crop_roundtrip_"+handedness, {"max_error_px": error},
               error < thresholds["roundtrip_px"], "invariant")
        raw["crop_"+handedness] = crop
    wrongly_recovered = (image_h @ (mirror@A).T @ mirror.T) @ np.linalg.inv(mirror@A).T
    record("double_reflection", {"mean_error_px": float(np.linalg.norm(wrongly_recovered[:, :2]-uv, axis=1).mean())},
           classification="injected_fault")

    R = Rotation.from_rotvec([.2, -.1, .05]).as_matrix()
    c = np.array([.01, -.02, .03])
    offset = np.array([.012, .006, -.008])
    # J = R*(p-offset) + offset + t. Transform the complete object by (Q,c).
    Q = Rotation.from_rotvec([-.15, .08, .3]).as_matrix()
    before = (local-offset)@R.T + offset + true_t
    t_new = Q@(offset+true_t)+c-offset
    after = (local-offset)@(Q@R).T+offset+t_new
    root_error = float(np.max(np.linalg.norm(after-(before@Q.T+c), axis=1)))
    record("root_offset_transform", {"max_error_m": root_error},
           root_error < thresholds["root_transform_m"], "invariant")
    record("root_offset_omitted", {"error_m": float(np.linalg.norm(t_new-(Q@true_t+c)))},
           classification="injected_fault")

    # Fixed synthetic output; camera postprocessing only, not a network test.
    s, tx, ty = cfg["weak_camera"]
    W, H = cfg["full_image_size_px"]
    for fcfg in cfg["virtual_focal_cfg"]:
        f = fcfg/side*max(W, H)
        z = 2*f/(cfg["bbox_size_px"]*s)
        trans = np.array([tx+(cx-W/2)*z/f, ty+(cy-H/2)*z/f, z])
        virtual_K = np.array([[f, 0, W/2], [0, f, H/2], [0, 0, 1.]])
        target = project(local+trans, virtual_K)
        fitted, condition, _ = translation_fit(local, target, K, np.ones(len(local)), cfg)
        reproj = float(np.linalg.norm(project(local+fitted, K)-target, axis=1).mean())
        raw[f"virtual_{int(fcfg)}_uv"] = target
        record(f"focal_decode_{int(fcfg)}",
               {"full_focal_px": f, "virtual_depth_m": z,
                "real_K_fit_translation_m": fitted.tolist(),
                "real_K_fit_residual_px": reproj, "condition": condition})

    sol_trace = []
    sol = fit_shared_translation(local, stereo, cameras, np.ones(stereo.shape[:2]),
                                 np.array([0., 0., 1.2]), cfg, True, sol_trace)
    traces["shared_translation_noiseless"] = sol_trace
    error = float(np.linalg.norm(sol.x-true_t))
    record("shared_translation_noiseless", {"error_m": error, "nfev": sol.nfev,
           "residual_calls": len(sol_trace), "solver_success": bool(sol.success)},
           bool(sol.success and error < thresholds["noiseless_translation_m"]), "invariant")

    f = K[0, 0]
    for z in cfg["distance_sweep_m"]:
        noise = rng.normal(0, cfg["per_view_horizontal_noise_sigma_px"], (cfg["noise_trials"], 2))
        disparity = f*base/z + noise[:, 0]-noise[:, 1]
        reconstructed = f*base/disparity
        raw[f"noise_{z}_per_view_px"] = noise
        raw[f"noise_{z}_depth_m"] = reconstructed
        observed = float(np.std(reconstructed-z, ddof=1))
        expected = z*z/(f*base)*np.sqrt(2)*cfg["per_view_horizontal_noise_sigma_px"]
        record(f"noise_depth_{z}", {"mean_bias_mm": float(np.mean(reconstructed-z)*1000),
               "std_mm": observed*1000, "linear_prediction_std_mm": float(expected*1000),
               "p95_absolute_error_mm": float(np.percentile(np.abs(reconstructed-z), 95)*1000)},
               bool(abs(observed/expected-1) < thresholds["noise_sigma_relative_tolerance"]), "statistical_check")
        for kind, fractions in [("focal", cfg["focal_perturbation_fraction"]),
                                ("baseline", cfg["baseline_perturbation_fraction"])]:
            for delta in fractions:
                measured = f*(1+delta)*base/(f*base/z)
                record(f"{kind}_{delta}_{z}", {"depth_error_mm": float((measured-z)*1000)},
                       classification="injected_fault")
        for frames in cfg["sync_offset_frames"]:
            delta_x = cfg["horizontal_velocity_m_s"]*frames/cfg["fps"]
            d = f*(base-delta_x)/z
            record(f"sync_{frames}_{z}", {"depth_error_mm": float((f*base/d-z)*1000),
                   "offset_ms": frames/cfg["fps"]*1000}, classification="injected_fault")

    corrupted = stereo.copy()
    corrupted[cfg["outlier_view"], cfg["outlier_joint_indices"]] += cfg["outlier_offset_px"]
    raw["outlier_uv_px"] = corrupted
    outlier_errors = {}
    for robust in [False, True]:
        name = "outlier_huber" if robust else "outlier_l2"
        trace = []
        fit = fit_shared_translation(local, corrupted, cameras, np.ones(stereo.shape[:2]),
                                     true_t + [0, 0, .2], cfg, robust, trace)
        traces[name] = trace
        outlier_errors[name] = float(np.linalg.norm(fit.x-true_t))
        record(name, {"translation_error_mm": outlier_errors[name]*1000,
                     "translation_m": fit.x.tolist(), "nfev": fit.nfev,
                     "residual_calls": len(trace), "solver_success": bool(fit.success)},
               bool(fit.success), "injected_fault")
    ratio = outlier_errors["outlier_huber"]/outlier_errors["outlier_l2"]
    record("robust_comparison", {"error_ratio": ratio},
           ratio < thresholds["robust_error_ratio_max"], "controlled_comparison")

    for fraction in cfg["missing_fraction"]:
        weights = np.ones(stereo.shape[:2])
        weights.ravel()[rng.choice(weights.size, round(weights.size*fraction), replace=False)] = 0
        trace = []
        fit = fit_shared_translation(local, stereo, cameras, weights, true_t+[0, 0, .2], cfg, True, trace)
        traces[f"missing_{fraction}"] = trace
        raw[f"missing_{fraction}_weights"] = weights
        error = float(np.linalg.norm(fit.x-true_t))
        record(f"missing_{fraction}", {"error_m": error, "valid_per_view": weights.sum(axis=1).tolist()},
               bool(fit.success and error < thresholds["noiseless_translation_m"]), "invariant")

    invalid_cases = {
        "all_missing": (local, np.full_like(uv, np.nan), np.ones(len(local))),
        "rank_deficient": (np.zeros_like(local), np.tile(Kg[:2, 2], (len(local), 1)), np.ones(len(local))),
        "negative_depth_solution": (local, (np.c_[local[:, :2]+true_t[:2], local[:, 2]-1] @ Kg.T), np.ones(len(local))),
    }
    p, h, w = invalid_cases["negative_depth_solution"]
    invalid_cases["negative_depth_solution"] = p, h[:, :2]/h[:, 2:], w
    for name, (points, target, weights) in invalid_cases.items():
        try:
            translation_fit(points, target, Kg, weights, cfg)
            record(name, {"rejection": None}, False, "expected_rejection")
        except ValueError as exc:
            record(name, {"rejection": str(exc)}, True, "expected_rejection")
    partial = uv.copy()
    partial[0] = np.nan
    fitted, _, valid = translation_fit(local, partial, Kg, np.ones(len(local)), cfg)
    record("partial_nan_mask", {"valid_points": int(valid.sum()),
           "error_m": float(np.linalg.norm(fitted-true_t))},
           bool(valid.sum() == len(local)-1 and np.linalg.norm(fitted-true_t) < 1e-9), "invariant")
    # Algebraic mono scale ambiguity: scaling both shape and translation leaves uv unchanged.
    scaled = project(1.2*(local+true_t), K)
    scale_residual = float(np.max(np.linalg.norm(scaled-stereo[0], axis=1)))
    right_scaled = project(1.2*(local+true_t)+cameras[1][2], K)
    record("monocular_scale_ambiguity", {"left_max_residual_px": scale_residual,
           "right_mean_residual_px": float(np.linalg.norm(right_scaled-stereo[1], axis=1).mean()),
           "translation_error_mm": float(np.linalg.norm(.2*true_t)*1000)},
           scale_residual < thresholds["roundtrip_px"], "invariant")
    # Equivalent objectives when view counts differ: per-view vs global pooled.
    per_view = .5*(.5*1**2 + .5*3**2)
    global_pooled = (21*.5*1**2+3*.5*3**2)/24
    record("view_normalization_counterexample", {"per_view_l2": per_view,
           "global_pooled_l2": global_pooled}, classification="counterexample")

    np.savez_compressed(out / "raw_arrays.npz", **raw)
    json_write(out / "solver_traces.json", traces)
    json_write(out / "results.json", results)
    with (out / "metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case", "classification", "passed", "metric", "value"])
        for item in results:
            for key, value in item["metrics"].items():
                writer.writerow([item["case"], item["classification"], item["passed"], key, json.dumps(value)])
    checks = [r for r in results if r["passed"] is not None]
    manifest.update(end_utc=datetime.now(timezone.utc).isoformat(), elapsed_s=time.perf_counter()-start,
                    cases=len(results), checks=len(checks), failed_checks=sum(not r["passed"] for r in checks))
    json_write(out / "manifest.json", manifest)
    json_write(out / "sha256.json", {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    print("RUN_DIRECTORY=" + str(out))
    return int(manifest["failed_checks"] > 0)


if __name__ == "__main__":
    sys.exit(main())
