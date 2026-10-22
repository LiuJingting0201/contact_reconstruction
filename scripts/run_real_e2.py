"""Real stereo HaMeR/MANO matrix; fail closed before inference on invalid data.

Usage: CUDA_VISIBLE_DEVICES=1 .venv-gpu/bin/python scripts/run_real_e2.py
       --manifest inputs/real_e2/manifest.json --run-id real_e2_unique_id
Validation only: append --validate-only. No synthetic fallback is implemented.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import cv2
import numpy as np
from real_e2_data import load_manifest, require, sha256, transform

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def seal(out):
    files = {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*"))
             if p.is_file() and p.name != "sha256.json"}
    write_json(out / "sha256.json", files)


def torch_setup():
    # Import existing helpers only: main() is guarded and no old run is executed.
    from run_mano_backend import MANO, tensor, array, geometry, rot6d_to_rotmat, CfgNode
    import torch
    from hamer.models.hamer import HAMER
    return torch, MANO, tensor, array, geometry, rot6d_to_rotmat, CfgNode, HAMER


def confidence_for(sample, kind):
    if kind == "uniform":
        return np.ones((2, 21), np.float32), np.ones(2, np.float32)
    values, view_weights = [], []
    for view in sample["row"]["views"]:
        if kind == "oracle":
            # Annotation validity is NOT oracle visibility.
            mask = view.get("oracle_visibility")
            require(mask is not None and view.get("oracle_visibility_source"), "oracle visibility unavailable")
            value = np.asarray(mask, np.float32)
            view_weight = 1.
        else:
            spec = view.get("predicted_confidence")
            require(spec is not None, "HaMeR has no confidence head; external prediction required for E2-5")
            require(spec["uses_evaluation_gt"] is False and spec["model"] and spec["checkpoint_sha256"],
                    "confidence provenance missing or uses evaluation GT")
            value = np.asarray(spec["joint_confidence"], np.float32)
            value *= np.asarray(spec["visibility"], np.float32)
            view_weight = float(spec["view_confidence"])
        require(value.shape == (21,) and np.isfinite(value).all() and (value >= 0).all()
                and (value <= 1).all(), "invalid confidence/mask")
        require(np.isfinite(view_weight) and 0 <= view_weight <= 1, "invalid view confidence")
        values.append(value)
        view_weights.append(view_weight)
    return np.stack(values), np.asarray(view_weights, np.float32)


def geometric_confidence(sample, pseudo, T):
    """Non-GT heuristic: epipolar agreement and image-bound visibility only.

    It cannot identify every coherent bad view; it is not learned visibility.
    Five pixels is fixed before seeing any optimization/evaluation result.
    """
    K = sample["K"]
    relative = T[1] @ np.linalg.inv(T[0])
    x, y, z = relative[:3, 3]
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0.]])
    F = np.linalg.inv(K[1]).T @ skew @ relative[:3, :3] @ np.linalg.inv(K[0])
    q = np.concatenate([pseudo, np.ones((2, 21, 1))], axis=-1)
    lines_right = q[0] @ F.T
    lines_left = q[1] @ F
    numerator = abs(np.sum(q[1] * lines_right, axis=-1))
    distances = np.stack([numerator/np.maximum(np.linalg.norm(lines_left[:, :2], axis=-1), 1e-9),
                          numerator/np.maximum(np.linalg.norm(lines_right[:, :2], axis=-1), 1e-9)])
    confidence = 1/(1+(distances/5.)**2)
    for v, view in enumerate(sample["views"]):
        height, width = view["image"].shape[:2]
        inside = (pseudo[v, :, 0] >= 0) & (pseudo[v, :, 0] < width)
        inside &= (pseudo[v, :, 1] >= 0) & (pseudo[v, :, 1] < height)
        confidence[v] *= inside
    return confidence.astype(np.float32), np.ones(2, np.float32)


def infer_teacher(model, cfg, sample, tensor, array, config):
    """One MANO inference per view, nominal pseudo + real-K camera interpretations."""
    import torch
    predictions = []
    for view in sample["views"]:
        A, K = view["A"], view["K"]
        crop = cv2.warpAffine(view["image"], A[:2], (256, 256))
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        normalized = (rgb.astype(np.float32)/255 - [.485, .456, .406]) / [.229, .224, .225]
        cfg.EXTRA.FOCAL_LENGTH = config["nominal_focal_crop_px"]
        with torch.no_grad():
            result = model.forward_step({"img": tensor(normalized.transpose(2, 0, 1)[None])})
        params = result["pred_mano_params"][0]
        def value(x):
            return array(x) if torch.is_tensor(x) else np.asarray(x)
        rotations = np.concatenate([value(params["global_orient"]).reshape(1, 1, 3, 3),
                                    value(params["hand_pose"]).reshape(1, 15, 3, 3)], axis=1)
        beta = value(params["betas"]).reshape(1, 10)
        xyz = value(result["pred_keypoints_3d"])[0].copy()
        weak = value(result["pred_cam"])[0]
        require(np.isfinite(weak).all() and weak[0] > 0, "nonpositive weak camera scale")
        pseudo_crop = (value(result["pred_keypoints_2d"])[0] + .5) * 256
        pseudo = np.c_[pseudo_crop, np.ones(21)] @ np.linalg.inv(A).T
        scale = abs(A[0, 0])
        # Anisotropic real K: geometric mean gives a documented scalar weak-camera
        # interpretation, not a claim that weak camera solves perspective depth.
        focal = np.sqrt(K[0, 0] * K[1, 1]) * scale
        t = np.array([weak[1], weak[2], 2 * focal / (256 * weak[0] + 1e-9)])
        if sample["row"]["handedness"] == "left":
            xyz[:, 0] *= -1
            t[0] *= -1
        center = np.linalg.inv(A) @ [128, 128, 1]
        t[:2] += (center[:2] - K[:2, 2]) * t[2] / K.diagonal()[:2]
        require((xyz[:, 2] + t[2]).min() > config["minimum_depth_m"], "invalid real-K initialization")
        predictions.append({"rotations": rotations, "betas": beta, "xyz": xyz,
                            "translation": t, "weak_camera": weak, "pseudo": pseudo[:, :2],
                            "nominal_camera_translation": value(result["pred_cam_t"])[0],
                            "real_focal_crop": focal, "affine": A})
    return predictions


def project_torch(xyz, K, T):
    import torch
    camera = torch.einsum("vab,jb->vja", T[:, :3, :3], xyz) + T[:, None, :3, 3]
    homogeneous = torch.einsum("vab,vjb->vja", K, camera)
    return homogeneous[..., :2] / homogeneous[..., 2:].clamp_min(1e-6), camera


def reprojection_torch(uv, target, confidence, weights, robust, scale):
    import torch
    active = confidence.sum(-1) > 0
    weight = weights * active
    require(float(weight.sum()) > 0, "no active pseudo observations")
    residual = torch.where(confidence[..., None] > 0, uv - target, torch.zeros_like(uv))
    d2 = (residual / scale).square().sum(-1)
    cost = torch.where(d2 <= 1, d2, 2*d2.clamp_min(1e-12).sqrt()-1) if robust else d2
    per_view = (cost * confidence).sum(-1) / confidence.sum(-1).clamp_min(1e-12)
    return (per_view * weight).sum() / weight.sum()


def solve(mano, initial, K, T, pseudo, confidence, view_confidence, variant, config, handedness, out):
    torch, _, tensor, array, geometry, decode, _, _ = torch_setup()
    r0, b0, t0 = map(tensor, initial)
    trans = torch.nn.Parameter(t0.clone())
    six = r0[..., :2].transpose(-1, -2).reshape(1, 16, 6)
    require(torch.allclose(decode(six).reshape_as(r0), r0, atol=1e-6), "6D rotation roundtrip")
    glob = torch.nn.Parameter(six[:, :1].clone())
    hand = torch.nn.Parameter(six[:, 1:].clone())
    beta = torch.nn.Parameter(b0.clone())
    mode = variant["variables"]
    groups = [{"params": [trans], "lr": config["translation_lr"]}]
    if mode in ("orientation", "pose", "shape"):
        groups.append({"params": [glob], "lr": config["rotation_shape_lr"]})
    if mode in ("pose", "shape"):
        groups.append({"params": [hand], "lr": config["rotation_shape_lr"]})
    if mode == "shape":
        groups.append({"params": [beta], "lr": config["rotation_shape_lr"]})
    opt = torch.optim.Adam(groups)
    K, T, pseudo, confidence = map(tensor, (K, T, pseudo, confidence))
    weights = tensor([float(v in variant["views"])*view_confidence[v] for v in range(2)])
    mirror = tensor([-1 if handedness == "left" else 1, 1, 1])
    def forward():
        g = decode(glob).reshape(1, 1, 3, 3) if mode in ("orientation", "pose", "shape") else r0[:, :1]
        h = decode(hand).reshape(1, 15, 3, 3) if mode in ("pose", "shape") else r0[:, 1:]
        r = torch.cat([g, h], 1)
        b = beta if mode == "shape" else b0
        xyz = geometry(mano, r, b)[0] * mirror + trans[0]
        uv, camera = project_torch(xyz, K, T)
        reproj = reprojection_torch(uv, pseudo, confidence, weights, variant["loss"] == "huber",
                                   config["reprojection_scale_px"])
        pose_prior = (r-r0).square().mean()
        shape_prior = (b-b0).square().mean()
        depth = (config["minimum_depth_m"]-camera[..., 2]).clamp_min(0).square().mean()
        loss = reproj + config["pose_prior_weight"]*pose_prior + config["shape_prior_weight"]*shape_prior
        loss = loss + config["depth_prior_weight"]*depth
        return r, b, xyz, uv, camera, loss, reproj, pose_prior, shape_prior, depth
    steps = 0 if mode == "none" else config["steps"]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    calls, updates, initial_loss, failure, trace = 0, 0, None, None, []
    for step in range(steps + 1):
        state = forward()
        calls += 1
        r, b, xyz, uv, camera, loss, reproj, pp, sp, dp = state
        if not torch.isfinite(loss):
            failure = f"nonfinite_objective_at_step_{step}"
            break
        if initial_loss is None:
            initial_loss = float(loss.detach())
        if step % config["trace_interval"] == 0 or step == steps:
            trace.append({"step": step, "objective": float(loss.detach()),
                          "reprojection": float(reproj.detach()), "pose_prior": float(pp.detach()),
                          "shape_prior": float(sp.detach()), "depth_penalty": float(dp.detach())})
        if step == steps:
            break
        opt.zero_grad()
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for g in groups for p in g["params"]):
            failure = f"nonfinite_gradient_at_step_{step}"
            break
        if step == config["decay_step"]:
            for g in opt.param_groups:
                g["lr"] *= config["decay_factor"]
        opt.step()
        updates += 1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated()
    out.mkdir(exist_ok=False)
    result = {"xyz": array(xyz), "uv": array(uv), "camera": array(camera), "rotations": array(r),
              "betas": array(b), "translation": array(trans), "objective": array(loss),
              "initial_objective": np.array(initial_loss), "K": array(K), "T": array(T)}
    np.savez_compressed(out/"state.npz", **result)
    torch.save({"translation": trans.detach(), "global6": glob.detach(), "hand6": hand.detach(),
                "betas": beta.detach(), "optimizer": opt.state_dict(), "step": updates}, out/"optimizer.pt")
    # Read both checkpoint and arrays from disk, copy into parameters, recompute.
    saved = torch.load(out/"optimizer.pt", map_location="cuda")
    with torch.no_grad():
        for p, name in [(trans, "translation"), (glob, "global6"), (hand, "hand6"), (beta, "betas")]:
            p.copy_(saved[name])
        reloaded = forward()
    disk = np.load(out/"state.npz")
    reload_error = float(np.max(abs(array(reloaded[2])-disk["xyz"])))
    require(np.allclose(array(reloaded[2]), disk["xyz"], atol=1e-7, equal_nan=True), "reload mismatch")
    write_json(out/"runtime.json", {"backend_s": elapsed, "optimizer_steps": updates,
               "forward_objective_calls": calls, "closure_calls": 0, "peak_allocated_gpu_bytes": peak,
               "numerical_failure": failure, "reload_xyz_max_abs_m": reload_error if np.isfinite(reload_error) else None,
               "timing_scope": "per sample backend only; excludes inference, serialization, reload; not FPS",
               "trajectory": trace})
    return result


def run(args, out, config):
    manifest, samples, validation = load_manifest(args.manifest, config, out/"gt_overlays")
    write_json(out/"validation.json", {"passed": True, "rows": validation,
               "visual_sync_review": manifest.get("visual_sync_review", "not supplied")})
    if args.validate_only:
        return {"status": "validated_only", "real_e2_executed": False}
    require(manifest.get("visual_sync_review", {}).get("passed") is True,
            "paired GT overlays require documented visual alignment/time review before inference")
    torch, _, tensor, array, geometry, _, CfgNode, HAMER = torch_setup()
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = CfgNode.load_cfg((ROOT/"inputs/model_config.yaml").read_text())
    cfg.MANO.DATA_DIR = str(ROOT/"inputs/data")
    cfg.MANO.MODEL_PATH = str(ROOT/"inputs/data/mano")
    cfg.MANO.MEAN_PARAMS = str(ROOT/"inputs/data/mano_mean_params.npz")
    cfg.MODEL.BBOX_SHAPE = [192, 256]
    cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS", None)
    model = HAMER(cfg, init_renderer=False)
    checkpoint = torch.load(ROOT/"inputs/hamer.ckpt", map_location="cpu")
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
    require(not any(not k.startswith(("discriminator.", "renderer.", "mesh_renderer."))
                    for k in incompatible.missing_keys+incompatible.unexpected_keys), "checkpoint mismatch")
    del checkpoint
    model = model.cuda().eval().requires_grad_(False)
    mano = model.mano
    write_json(out/"teacher_provenance.json", {"checkpoint_sha256": sha256(ROOT/"inputs/hamer.ckpt"),
               "branches": "same MANO prediction, two camera interpretations; real-K depth is not GT",
               "confidence": "HaMeR has no learned confidence output",
               "anisotropic_K_rule": "geometric mean of crop fx,fy for scalar weak-camera depth"})
    (out/"teacher_config.yaml").write_text(cfg.dump())
    status = []
    for i, sample in enumerate(samples):
        sample_out = out/f"sample_{i:03d}"
        sample_out.mkdir()
        write_json(sample_out/"sample.json", sample["row"])
        teacher = infer_teacher(model, cfg, sample, tensor, array, config)
        np.savez_compressed(sample_out/"teacher.npz",
                            **{f"{v}_{k}": x for v, p in enumerate(teacher) for k, x in p.items()})
        left = teacher[0]
        with torch.no_grad():
            reconstructed = array(geometry(mano, tensor(left["rotations"]), tensor(left["betas"])))[0]
        reconstructed[:, 0] *= -1 if sample["row"]["handedness"] == "left" else 1
        require(np.allclose(reconstructed, left["xyz"], atol=1e-6), "MANO/teacher reconstruction mismatch")
        # The shared state lives in the LEFT camera coordinates.
        T = sample["T"] @ np.linalg.inv(sample["T"][0])
        gt_left = sample["gt_camera"][0]
        initial = (left["rotations"], left["betas"], left["translation"][None])
        pseudo = np.stack([p["pseudo"] for p in teacher])
        np.savez_compressed(sample_out/"evaluation.npz", gt=gt_left, gt_uv=sample["gt_uv"],
                            gt_camera=sample["gt_camera"],
                            right_gt_in_left=transform(sample["gt_camera"][1], np.linalg.inv(T[1])),
                            valid=sample["valid"], pseudo=pseudo, K=sample["K"], T=T,
                            affine=np.stack([p["affine"] for p in teacher]),
                            rotations=initial[0], betas=initial[1], translation=initial[2])
        for variant in config["variants"]:
            label = variant["id"]
            try:
                confidence, view_confidence = (
                    geometric_confidence(sample, pseudo, T) if variant["confidence"] == "geometry"
                    else confidence_for(sample, variant["confidence"]))
            except ValueError as error:
                status.append({"sample": i, "variant": label, "status": "unavailable", "reason": str(error)})
                continue
            np.savez_compressed(sample_out/f"{label}_confidence.npz", joint=confidence, view=view_confidence)
            final = solve(mano, initial, sample["K"], T, pseudo, confidence, view_confidence, variant, config,
                          sample["row"]["handedness"], sample_out/label)
            tiles = []
            for v, view in enumerate(sample["views"]):
                tile = view["image"].copy()
                for points, color in [(sample["gt_uv"][v], (0, 255, 0)), (pseudo[v], (255, 0, 0)),
                                      (final["uv"][v], (0, 0, 255))]:
                    for point in points:
                        if np.isfinite(point).all() and (abs(point) < 1e6).all():
                            cv2.circle(tile, tuple(np.rint(point).astype(int)), 2, color, -1)
                tiles.append(tile)
            height = max(t.shape[0] for t in tiles)
            tiles = [cv2.copyMakeBorder(t, 0, height-t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
            cv2.imwrite(str(sample_out/label/"overlay.jpg"), np.concatenate(tiles, axis=1))
            status.append({"sample": i, "variant": label, "status": "completed"})
            print(json.dumps(status[-1]), flush=True)
        write_json(out/"matrix_status.json", status)
    return {"status": "matrix_finished", "real_e2_executed": True, "samples": len(samples),
            "synchronization_ablation": "not executed: requires explicitly supplied neighboring right frames",
            "E3": "closed pending independent real-E2 evaluation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/real_e2_v1.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    require(Path(args.run_id).name == args.run_id and args.run_id not in (".", ".."), "invalid run_id")
    out = ROOT/"artifacts"/args.run_id
    out.mkdir(exist_ok=False)
    config = json.loads(args.config.read_text())
    write_json(out/"config.json", config)
    shutil.copy2(args.manifest, out/"input_manifest.json")
    for file in ("run_real_e2.py", "real_e2_data.py", "evaluate_real_e2.py"):
        shutil.copy2(ROOT/"scripts"/file, out/file)
    git = subprocess.run(["git", "diff", "--binary"], cwd=ROOT, text=True, capture_output=True)
    (out/"git_diff.txt").write_text(git.stdout if git.returncode == 0 else git.stderr)
    try:
        result = run(args, out, config)
        write_json(out/"status.json", result)
        if result["real_e2_executed"]:
            subprocess.run([os.sys.executable, str(ROOT/"scripts/evaluate_real_e2.py"), str(out)], check=True)
    except Exception:
        (out/"failure.txt").write_text(traceback.format_exc())
        write_json(out/"status.json", {"status": "failed; retain partial outputs",
                   "real_e2_completed": False, "E3": "closed"})
        raise
    finally:
        seal(out)


if __name__ == "__main__":
    main()
