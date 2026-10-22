"""Run the one-shot ResNet raw/B2 E3 evaluation on the frozen 32-sample set."""
import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_pose_preserving_e2 import one, stats
from real_e2_data import require, sha256
from run_real_e2 import solve
from train_resnet50_rhd_full import configure_model


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def project(xyz, K, T):
    camera = np.einsum("vab,jb->vja", T[:, :3, :3], xyz) + T[:, None, :3, 3]
    homogeneous = np.einsum("vab,vjb->vja", K, camera)
    return homogeneous[..., :2] / homogeneous[..., 2:], camera


def load_crop(view, affine):
    image = cv2.imread(view["image"])
    require(image is not None and image.shape[:2] == (480, 640), "missing/malformed E3 RGB")
    require(sha256(view["image"]) == view["image_sha256"], "E3 RGB hash changed")
    crop = cv2.warpAffine(image, affine[:2], (256, 256), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    normalized = (rgb - np.asarray([.485, .456, .406], np.float32)) / np.asarray(
        [.229, .224, .225], np.float32)
    return torch.from_numpy(normalized.transpose(2, 0, 1)).float(), crop


def infer_view(model, image, affine, K, device):
    with torch.no_grad():
        output = model.forward_step({"img": image[None].to(device)})
    params = output["pred_mano_params"]
    rotations = np.concatenate([
        params["global_orient"].detach().cpu().numpy().reshape(1, 1, 3, 3),
        params["hand_pose"].detach().cpu().numpy().reshape(1, 15, 3, 3),
    ], axis=1)
    betas = params["betas"].detach().cpu().numpy().reshape(1, 10)
    relative_xyz = output["pred_keypoints_3d"][0].detach().cpu().numpy()
    weak = output["pred_cam"][0].detach().cpu().numpy()
    require(np.isfinite(weak).all() and weak[0] > 0, "invalid ResNet weak camera")
    crop_uv = (output["pred_keypoints_2d"][0].detach().cpu().numpy() + .5) * 256
    pseudo = np.c_[crop_uv, np.ones(21)] @ np.linalg.inv(affine).T
    focal = np.sqrt(K[0, 0] * K[1, 1]) * abs(affine[0, 0])
    translation = np.asarray([weak[1], weak[2], 2 * focal / (256 * weak[0] + 1e-9)])
    center = np.linalg.inv(affine) @ [128, 128, 1]
    translation[:2] += (center[:2] - K[:2, 2]) * translation[2] / K.diagonal()[:2]
    require(np.isfinite(relative_xyz).all() and
            (relative_xyz[:, 2] + translation[2]).min() > .01,
            "invalid ResNet real-K initialization")
    return {
        "rotations": rotations, "betas": betas, "relative_xyz": relative_xyz,
        "translation": translation, "weak_camera": weak, "pseudo": pseudo[:, :2],
        "real_focal_crop": focal,
    }


def benchmark(model, image, device):
    model.eval()
    with torch.no_grad():
        for _ in range(20):
            model.forward_step({"img": image[None].to(device)})
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        values = []
        for _ in range(100):
            start = time.perf_counter()
            model.forward_step({"img": image[None].to(device)})
            torch.cuda.synchronize(device)
            values.append((time.perf_counter() - start) * 1000)
    return {
        "scope": "batch=1 network+MANO only; 20 warmup, 100 timed, CUDA synchronized",
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def aggregate(rows):
    keys = rows[0]["metrics"]
    return {key: stats([row["metrics"][key] for row in rows]) for key in keys}


def metric_mean(aggregates, variant, key):
    return aggregates[variant][key]["mean"]


def frontend_metrics(pseudo, target):
    valid = target["valid"].astype(bool)
    epe = np.linalg.norm(pseudo - target["gt_uv"], axis=-1)
    spans = np.ptp(pseudo, axis=1).max(axis=1)
    return {
        "left_frontend_gt2d_epe_px": float(epe[0][valid[0]].mean()),
        "right_frontend_gt2d_epe_px": float(epe[1][valid[1]].mean()),
        "left_frontend_projection_span_px": float(spans[0]),
        "right_frontend_projection_span_px": float(spans[1]),
        "frontend_projection_collapse": float((spans < 5).any()),
        "frontend_nonfinite": float(not np.isfinite(pseudo).all()),
    }


def cached_vit_frontend(source):
    rows = []
    for folder in sorted(source.glob("sample_*")):
        target = np.load(folder / "evaluation.npz")
        rows.append(frontend_metrics(target["pseudo"], target))
    return {key: stats([row[key] for row in rows]) for key in rows[0]}


def make_report(result):
    vit, res = result["vit_cached"], result["resnet"]
    metrics = [
        "absolute_mpjpe_mm", "wrist_relative_mpjpe_mm",
        "rotation_aligned_wrist_relative_mpjpe_mm_diagnostic",
        "left_wrist_depth_error_mm", "right_wrist_depth_error_mm",
        "wrist_position_error_mm", "fingertip_absolute_error_mm",
        "fingertip_wrist_relative_error_mm", "left_gt2d_epe_px",
        "right_gt2d_epe_px", "pseudo_residual_px", "quality_failure",
        "numerical_failure", "positive_depth_failure",
    ]
    lines = [
        "# Fixed E3: ViT versus ResNet-50 with frozen B2",
        "",
        "The ResNet checkpoint was selected only on the RHD development split. "
        "The 32 final DexYCB pairs were evaluated once after checkpoint freeze.",
        "",
        "| Metric | ViT raw | ViT+B2 | ResNet raw | ResNet+B2 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key in metrics:
        lines.append(
            f"| {key} | {metric_mean(vit, 'B0', key):.3f} | "
            f"{metric_mean(vit, 'B2', key):.3f} | "
            f"{metric_mean(res, 'raw', key):.3f} | "
            f"{metric_mean(res, 'B2', key):.3f} |"
        )
    lines += [
        "",
        "### Independent per-view frontend 2D",
        "",
        "| Metric | ViT | ResNet |",
        "| --- | ---: | ---: |",
    ]
    for key in ("left_frontend_gt2d_epe_px", "right_frontend_gt2d_epe_px",
                "left_frontend_projection_span_px", "right_frontend_projection_span_px",
                "frontend_projection_collapse", "frontend_nonfinite"):
        lines.append(f"| {key} | {vit['frontend'][key]['mean']:.3f} | "
                     f"{res['frontend'][key]['mean']:.3f} |")
    lines += [
        "",
        "## Compensation",
        "",
        f"- Raw degradation: {result['compensation']['raw_degradation_mm']:.3f} mm",
        f"- Optimized degradation: {result['compensation']['optimized_degradation_mm']:.3f} mm",
        f"- Backend recovery: {result['compensation']['backend_recovery_mm']:.3f} mm",
        f"- Compensation ratio: {result['compensation']['compensation_ratio']}",
        "",
        "## Replacement decision",
        "",
        f"Decision: **{result['replacement']['decision']}**",
    ]
    for key, gate in result["replacement"]["gates"].items():
        lines.append(f"- {key}: {'PASS' if gate['passed'] else 'FAIL'} "
                     f"(value={gate['value']}, limit={gate['limit']})")
    lines += [
        "",
        "B2 changes translation only. Global orientation, articulated pose and shape "
        "are exact-equality frozen, so any remaining wrist-relative/fingertip error "
        "is frontend geometry that B2 cannot repair.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    out = ROOT / "artifacts" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    source = ROOT / "artifacts/dex_real_e2_matrix_20261020_02"
    baseline_run = ROOT / "artifacts/dex_pose_preserving_20261020_01"
    checkpoint_path = args.checkpoint.resolve()
    require(checkpoint_path.is_file(), "missing frozen best-dev checkpoint")

    config = json.loads((baseline_run / "config.json").read_text())
    variant = next(item for item in config["variants"] if item["id"] == "B2")
    require(variant == {"id": "B2", "variables": "translation", "views": [0, 1],
                        "loss": "huber", "lambda_rot": 0}, "B2 definition changed")
    require(config["nominal_focal_crop_px"] == 5000 and config["steps"] == 800 and
            config["translation_lr"] == .001 and config["reprojection_scale_px"] == 3,
            "frozen B2 hyperparameters changed")
    shutil.copy2(__file__, out / Path(__file__).name)
    shutil.copy2(baseline_run / "config.json", out / "frozen_b2_config.json")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    train_config = checkpoint["config"]
    require(train_config["teacher"] is None and not train_config["kd"] and
            not train_config["adversarial"], "checkpoint is not the frozen no-KD model")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    model, cfg, imagenet = configure_model(train_config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval().requires_grad_(False)
    require(float(cfg.EXTRA.FOCAL_LENGTH) == 5000 and int(cfg.MODEL.IMAGE_SIZE) == 256,
            "frontend nominal camera changed")

    raw_rows, b2_rows, frontend_rows, ordered_ids = [], [], [], []
    benchmark_image = None
    for index, source_folder in enumerate(sorted(source.glob("sample_*"))):
        sample_out = out / f"sample_{index:03d}"
        sample_out.mkdir()
        row = json.loads((source_folder / "sample.json").read_text())
        require(row["handedness"] == "right", "current frozen E3 contract is right-hand only")
        ordered_ids.append(row["sample_id"])
        write_json(sample_out / "sample.json", row)
        cached = np.load(source_folder / "evaluation.npz")
        K, T, affine = cached["K"], cached["T"], cached["affine"]
        predictions, crops = [], []
        for view_index, view in enumerate(row["views"]):
            image, crop = load_crop(view, affine[view_index])
            predictions.append(infer_view(model, image, affine[view_index],
                                          K[view_index], device))
            crops.append(crop)
            if benchmark_image is None:
                benchmark_image = image
        left = predictions[0]
        pseudo = np.stack([item["pseudo"] for item in predictions])
        initial = (left["rotations"], left["betas"], left["translation"][None])
        raw_xyz = left["relative_xyz"] + left["translation"]
        raw_uv, raw_camera = project(raw_xyz, K, T)
        raw_state = {
            "xyz": raw_xyz, "uv": raw_uv, "camera": raw_camera,
            "rotations": left["rotations"], "betas": left["betas"],
            "translation": left["translation"][None],
        }
        np.savez_compressed(sample_out / "predictions.npz",
                            **{f"{v}_{key}": value for v, item in enumerate(predictions)
                               for key, value in item.items()})
        np.savez_compressed(sample_out / "evaluation.npz",
                            gt=cached["gt"], gt_uv=cached["gt_uv"],
                            gt_camera=cached["gt_camera"], valid=cached["valid"],
                            pseudo=pseudo, K=K, T=T, affine=affine,
                            rotations=initial[0], betas=initial[1], translation=initial[2])
        raw_dir = sample_out / "raw"
        raw_dir.mkdir()
        np.savez_compressed(raw_dir / "state.npz", **raw_state)
        raw_runtime = {
            "backend_s": 0.0, "numerical_failure": None,
            "timing_scope": "raw frontend state; no backend",
        }
        write_json(raw_dir / "runtime.json", raw_runtime)
        target = np.load(sample_out / "evaluation.npz")
        per_view_metrics = frontend_metrics(pseudo, target)
        frontend_rows.append({"sample_id": row["sample_id"],
                              "metrics": per_view_metrics})
        raw_metrics = one(np.load(raw_dir / "state.npz"), target, raw_runtime, row, config)
        raw_rows.append({"sample_id": row["sample_id"], "variant": "raw",
                         "metrics": raw_metrics})

        confidence = np.ones((2, 21), np.float32)
        view_confidence = np.ones(2, np.float32)
        solve(model.mano, initial, K, T, pseudo, confidence, view_confidence,
              variant, config, row["handedness"], sample_out / "B2")
        b2_state = np.load(sample_out / "B2/state.npz")
        np.testing.assert_array_equal(b2_state["rotations"], initial[0])
        np.testing.assert_array_equal(b2_state["betas"], initial[1])
        b2_runtime = json.loads((sample_out / "B2/runtime.json").read_text())
        b2_metrics = one(b2_state, target, b2_runtime, row, config)
        b2_rows.append({"sample_id": row["sample_id"], "variant": "B2",
                        "metrics": b2_metrics})
        write_json(sample_out / "metrics.json",
                   {"frontend": per_view_metrics, "raw": raw_metrics, "B2": b2_metrics})
        print(json.dumps({"sample": index + 1, "total": 32, "id": row["sample_id"],
                          "raw_abs_mm": raw_metrics["absolute_mpjpe_mm"],
                          "b2_abs_mm": b2_metrics["absolute_mpjpe_mm"]}), flush=True)

    require(len(ordered_ids) == 32 and len(set(ordered_ids)) == 32,
            "final E3 sample set changed")
    expected_ids = [
        json.loads((source / f"sample_{index:03d}/sample.json").read_text())["sample_id"]
        for index in range(32)
    ]
    require(ordered_ids == expected_ids, "E3 sample order mismatch")
    resnet = {
        "frontend": {key: stats([row["metrics"][key] for row in frontend_rows])
                     for key in frontend_rows[0]["metrics"]},
        "raw": aggregate(raw_rows), "B2": aggregate(b2_rows),
    }
    vit_report = json.loads((baseline_run / "independent_metrics_final.json").read_text())
    vit = {key: vit_report["aggregates"][key] for key in ("B0", "B2")}
    vit["frontend"] = cached_vit_frontend(source)
    require(abs(metric_mean(vit, "B0", "absolute_mpjpe_mm") - 41.911) < .01 and
            abs(metric_mean(vit, "B2", "absolute_mpjpe_mm") - 8.125) < .01,
            "cached ViT reference changed")

    raw_degradation = (metric_mean(resnet, "raw", "absolute_mpjpe_mm") -
                       metric_mean(vit, "B0", "absolute_mpjpe_mm"))
    optimized_degradation = (metric_mean(resnet, "B2", "absolute_mpjpe_mm") -
                             metric_mean(vit, "B2", "absolute_mpjpe_mm"))
    backend_recovery = (metric_mean(resnet, "raw", "absolute_mpjpe_mm") -
                        metric_mean(resnet, "B2", "absolute_mpjpe_mm"))
    compensation_ratio = None if abs(raw_degradation) < 1e-6 else (
        1 - optimized_degradation / raw_degradation)
    stereo_2d_delta = np.mean([
        resnet["frontend"]["left_frontend_gt2d_epe_px"]["mean"] -
        vit["frontend"]["left_frontend_gt2d_epe_px"]["mean"],
        resnet["frontend"]["right_frontend_gt2d_epe_px"]["mean"] -
        vit["frontend"]["right_frontend_gt2d_epe_px"]["mean"],
    ])
    gates = {
        "absolute_mpjpe_increase_mm": {
            "value": optimized_degradation, "limit": max(2., .15 *
                metric_mean(vit, "B2", "absolute_mpjpe_mm")),
        },
        "wrist_relative_increase_mm": {
            "value": metric_mean(resnet, "B2", "wrist_relative_mpjpe_mm") -
                     metric_mean(vit, "B2", "wrist_relative_mpjpe_mm"),
            "limit": 2.,
        },
        "fingertip_wrist_relative_increase_mm": {
            "value": metric_mean(resnet, "B2", "fingertip_wrist_relative_error_mm") -
                     metric_mean(vit, "B2", "fingertip_wrist_relative_error_mm"),
            "limit": 3.,
        },
        "quality_failures": {
            "value": metric_mean(resnet, "B2", "quality_failure") * 32,
            "limit": 1.,
        },
        "stereo_mean_gt2d_epe_increase_px": {
            "value": float(stereo_2d_delta), "limit": 2.,
        },
        "projection_collapse_failures": {
            "value": 32 * resnet["frontend"]["frontend_projection_collapse"]["mean"],
            "limit": 0.0,
        },
        "numerical_or_depth_failures": {
            "value": 32 * (resnet["frontend"]["frontend_nonfinite"]["mean"] +
                           metric_mean(resnet, "B2", "numerical_failure") +
                           metric_mean(resnet, "B2", "positive_depth_failure")),
            "limit": 0.0,
        },
    }
    for gate in gates.values():
        gate["passed"] = bool(gate["value"] <= gate["limit"] + 1e-12)
    latency = benchmark(model, benchmark_image, device)
    result = {
        "protocol": "fixed_e3_resnet50_no_kd_b2_v1",
        "checkpoint": {
            "path": str(checkpoint_path), "sha256": sha256(checkpoint_path),
            "selected_step": checkpoint.get("best", {}).get("step"),
            "selected_epoch": checkpoint.get("best", {}).get("epoch"),
            "selection_used_final_e3": False,
            "initialization": "ImageNet1K V2 backbone; Fast-HaMeR MANO head trained from scratch",
            "teacher": None, "kd": False,
        },
        "sample_ids": ordered_ids,
        "sample_ids_sha256": hashlib.sha256("\n".join(ordered_ids).encode()).hexdigest(),
        "vit_cached": vit, "resnet": resnet,
        "rows": {"frontend": frontend_rows, "raw": raw_rows, "B2": b2_rows},
        "compensation": {
            "raw_degradation_mm": float(raw_degradation),
            "optimized_degradation_mm": float(optimized_degradation),
            "backend_recovery_mm": float(backend_recovery),
            "compensation_ratio": None if compensation_ratio is None else float(compensation_ratio),
            "undefined_rule": "undefined when abs(raw degradation) < 1e-6 mm",
        },
        "replacement": {
            "decision": "SUPPORTED" if all(gate["passed"] for gate in gates.values())
                        else "NOT_SUPPORTED",
            "gates": gates,
        },
        "frontend_latency": latency,
        "frozen_backend": {
            "definition": "calibrated stereo + Huber + translation only",
            "nominal_focal_crop_px": config["nominal_focal_crop_px"],
            "config_sha256": sha256(baseline_run / "config.json"),
        },
    }
    write_json(out / "results.json", result)
    (out / "results.md").write_text(make_report(result))
    write_json(out / "sha256.json", {
        str(path.relative_to(out)): sha256(path) for path in sorted(out.rglob("*"))
        if path.is_file() and path.name != "sha256.json"
    })
    print(json.dumps({"completed": True, "decision": result["replacement"]["decision"],
                      "results": str(out / "results.json")}), flush=True)


if __name__ == "__main__":
    main()
