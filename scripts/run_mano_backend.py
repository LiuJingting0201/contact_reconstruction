"""Frozen-frontend E1 and explicitly synthetic E2 controls; no network training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
for name, value in {"bool": np.bool_, "int": int, "float": float, "complex": complex,
                    "object": object, "unicode": str, "str": str}.items():
    if name not in np.__dict__:
        setattr(np, name, value)
import torch
from yacs.config import CfgNode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "repos/hamer"))
from hamer.models.mano_wrapper import MANO
from hamer.utils.geometry import rot6d_to_rotmat


def tensor(value):
    return torch.as_tensor(value, dtype=torch.float32, device="cuda")


def array(value):
    return value.detach().cpu().numpy()


def geometry(mano, rotations, betas):
    return mano(global_orient=rotations[:, :1], hand_pose=rotations[:, 1:],
                betas=betas, pose2rot=False).joints


def project(xyz, K, R, t):
    camera = torch.einsum("vab,njb->nvja", R, xyz) + t[None, :, None]
    homogeneous = torch.einsum("nvab,nvjb->nvja", K, camera)
    return homogeneous[..., :2] / homogeneous[..., 2:].clamp_min(1e-5), camera


def validate_views(K, R, t, confidence):
    assert torch.isfinite(K).all() and torch.isfinite(R).all() and torch.isfinite(t).all()
    assert (K[..., 0, 0] > 0).all() and (K[..., 1, 1] > 0).all()
    assert torch.allclose(R @ R.transpose(-1, -2), torch.eye(3, device="cuda"), atol=1e-5)
    assert torch.allclose(torch.linalg.det(R), torch.ones(len(R), device="cuda"), atol=1e-5)
    assert torch.isfinite(confidence).all() and (confidence >= 0).all()
    assert ((confidence > 0).sum(-1) >= 4).all()
    if len(R) > 1:
        centers = -torch.einsum("vba,vb->va", R, t)
        assert torch.linalg.vector_norm(centers[1] - centers[0]) > 1e-4


def solve(mano, out, label, initial, observations, K, R, ext_t, confidence,
          mode, robust=False, steps=800):
    validate_views(K, R, ext_t, confidence)
    rot0, beta0, trans0 = initial
    n = len(rot0)
    trans = torch.nn.Parameter(trans0.clone())
    # This HaMeR checkout stores two consecutive column vectors (2x3),
    # unlike the interleaved 3x2 layout used by some other implementations.
    initial6 = rot0[..., :2].transpose(-1, -2).reshape(n, 16, 6).contiguous()
    reconstructed = rot6d_to_rotmat(initial6).reshape(n, 16, 3, 3)
    assert torch.max(abs(reconstructed - rot0)) < 1e-6, "rotation_6d_roundtrip"
    global6 = torch.nn.Parameter(initial6[:, :1].clone())
    hand6 = torch.nn.Parameter(initial6[:, 1:].clone())
    beta = torch.nn.Parameter(beta0.clone())
    groups = [{"params": [trans], "lr": .001}]
    if mode != "translation":
        groups.append({"params": [global6], "lr": .01})
    if mode in ("translation_pose", "translation_pose_shape"):
        groups.append({"params": [hand6], "lr": .01})
    if mode == "translation_pose_shape":
        groups.append({"params": [beta], "lr": .01})
    optimizer = torch.optim.Adam(groups)
    trajectory = {k: [] for k in ("rotations", "betas", "translation", "objective")}
    begin = time.perf_counter()
    initial_objective = None
    history = []
    with (out / f"{label}_steps.jsonl").open("w") as log:
        for step in range(steps + 1):
            gr = rot0[:, :1] if mode == "translation" else rot6d_to_rotmat(global6).reshape(n, 1, 3, 3)
            hr = (rot6d_to_rotmat(hand6).reshape(n, 15, 3, 3)
                  if mode in ("translation_pose", "translation_pose_shape") else rot0[:, 1:])
            rotations = torch.cat([gr, hr], dim=1)
            betas = beta if mode == "translation_pose_shape" else beta0
            xyz = geometry(mano, rotations, betas) + trans[:, None]
            uv, camera = project(xyz, K, R, ext_t)
            d2 = ((uv - observations) / 3).square().sum(-1)
            # Huber of vector norm, scaled to agree with L2 in its quadratic region.
            point_cost = torch.where(d2 <= 1, d2, 2 * d2.clamp_min(1e-12).sqrt() - 1) if robust else d2
            reproj = ((point_cost * confidence).sum(-1) / confidence.sum(-1)).sum(-1)
            pose_prior = (rotations - rot0).square().mean((1, 2, 3))
            shape_prior = (betas - beta0).square().mean(1)
            depth_penalty = (.01 - camera[..., 2]).clamp_min(0).square().mean((1, 2))
            objective = reproj + .1 * pose_prior + .01 * shape_prior + 1e6 * depth_penalty
            assert torch.isfinite(objective).all(), (label, step, "nonfinite_objective")
            if initial_objective is None:
                initial_objective = array(objective)
            optimizer.zero_grad()
            objective.sum().backward()
            grad_norm = torch.sqrt(sum(p.grad.square().sum() for g in groups for p in g["params"]))
            assert torch.isfinite(grad_norm), (label, step, "nonfinite_gradient")
            current = array(objective)
            history.append(current)
            record = {"step": step, "objective": current.tolist(), "reprojection": array(reproj).tolist(),
                      "pose_prior": array(pose_prior).tolist(), "shape_prior": array(shape_prior).tolist(),
                      "depth_penalty": array(depth_penalty).tolist(), "gradient_norm": float(grad_norm),
                      "pseudo_residual_px": array((uv-observations).norm(dim=-1).mean((1, 2))).tolist()}
            log.write(json.dumps(record) + "\n")
            for key, value in [("rotations", rotations), ("betas", betas),
                               ("translation", trans), ("objective", objective)]:
                trajectory[key].append(array(value).copy())
            if step % 100 == 0:
                log.flush()
                print(json.dumps({"variant": label, "step": step, "mean_objective": float(objective.mean()),
                                  "elapsed_s": time.perf_counter() - begin}), flush=True)
            if step == steps:
                break
            if step == 600:
                for group in optimizer.param_groups:
                    group["lr"] *= .1
            optimizer.step()
    result = {"xyz": array(xyz), "uv": array(uv), "rotations": array(rotations),
              "betas": array(betas), "translation": array(trans), "objective": array(objective),
              "initial_objective": initial_objective,
              "minimum_depth": array(camera[..., 2].amin((1, 2))),
              "last100_objective_change": history[-1] - history[-101]}
    np.savez_compressed(out / f"{label}_trajectory.npz", **{k: np.stack(v) for k, v in trajectory.items()})
    np.savez_compressed(out / f"{label}_final.npz", **result)
    torch.save({"translation": trans.detach(), "global6": global6.detach(), "hand6": hand6.detach(),
                "betas": beta.detach(), "optimizer": optimizer.state_dict(), "step": steps},
               out / f"{label}_optimizer.pt")
    return result


def metrics(result, gt, gt_uv, visible, affine_scale, initial):
    xyz = result["xyz"]
    rot0, beta0, trans0 = [array(x) for x in initial]
    angles = np.arccos(np.clip((np.trace(result["rotations"] @ rot0.transpose(0, 1, 3, 2),
                                        axis1=-2, axis2=-1) - 1) / 2, -1, 1)) * 180 / np.pi
    uv_error = np.linalg.norm(result["uv"] - gt_uv, axis=-1)
    per_view = (uv_error * visible).sum(-1) / visible.sum(-1)
    rows = {
        "gt2d_full_px": per_view.mean(1),
        "gt2d_crop_px": (per_view * affine_scale).mean(1),
        "root_relative_mpjpe_mm": np.linalg.norm((xyz - xyz[:, :1]) - (gt - gt[:, :1]), axis=-1).mean(1) * 1000,
        "absolute_mpjpe_mm": np.linalg.norm(xyz - gt, axis=-1).mean(1) * 1000,
        "wrist_depth_error_mm": abs(xyz[:, 0, 2] - gt[:, 0, 2]) * 1000,
        "global_rotation_change_deg": angles[:, 0],
        "mean_hand_rotation_change_deg": angles[:, 1:].mean(1),
        "beta_l2_change": np.linalg.norm(result["betas"] - beta0, axis=1),
        "translation_change_mm": np.linalg.norm(result["translation"] - trans0, axis=1) * 1000,
    }
    numerical = (~np.isfinite(xyz).all((1, 2)) | (result["minimum_depth"] <= .01)
                 | (result["objective"] > result["initial_objective"] + 1e-4))
    quality = ((rows["gt2d_crop_px"] > 20) | (rows["root_relative_mpjpe_mm"] > 50)
               | (rows["wrist_depth_error_mm"] > 100))
    return {"mean": {k: float(v.mean()) for k, v in rows.items()},
            "numerical_failure_count": int(numerical.sum()), "quality_failure_count": int(quality.sum()),
            "numerical_failure_rate": float(numerical.mean()), "quality_failure_rate": float(quality.mean()),
            "per_image": {k: v.tolist() for k, v in rows.items()},
            "per_view_gt2d_px": per_view.tolist(),
            "numerical_failure": numerical.tolist(), "quality_failure": quality.tolist()}


def baseline(mano, initial, K, R, ext_t):
    rotations, betas, translation = initial
    with torch.no_grad():
        xyz = geometry(mano, rotations, betas) + translation[:, None]
        uv, camera = project(xyz, K, R, ext_t)
    return {"xyz": array(xyz), "uv": array(uv), "rotations": array(rotations),
            "betas": array(betas), "translation": array(translation),
            "minimum_depth": array(camera[..., 2].amin((1, 2))),
            "objective": np.zeros(len(xyz)), "initial_objective": np.zeros(len(xyz))}


def e1(mano, out):
    teacher_dir = ROOT / "artifacts/teacher_geometry_control_run02"
    source = np.load(teacher_dir / "outputs.npz")
    target_path = ROOT / "artifacts/rhd_supervised_v3/evaluation_targets.npz"
    target = np.load(target_path)
    raw_path = ROOT / "inputs/rhd/evaluation_subset.npz"
    raw = np.load(raw_path)
    ids = target["ids"]
    assert np.array_equal(ids, raw["ids"])
    report = json.loads((teacher_dir / "results.json").read_text())
    assert ids.tolist() == [r["id"] for r in report["rows"]]
    n = len(ids)
    rotations, betas, translations, pseudo = [], [], [], []
    for i in range(n):
        for key in ("global_orient", "hand_pose", "betas", "pred_cam", "pred_keypoints_3d"):
            assert np.allclose(source[f"{i}_nominal_{key}"], source[f"{i}_real_{key}"], atol=1e-6, rtol=0)
        rotations.append(np.concatenate([source[f"{i}_nominal_global_orient"], source[f"{i}_nominal_hand_pose"]]))
        betas.append(source[f"{i}_nominal_betas"])
        translations.append(source[f"{i}_initial_translation_m"])
        pseudo.append(source[f"{i}_pseudo_full"])
    initial = tuple(tensor(np.stack(x)) for x in (rotations, betas, translations))
    with torch.no_grad():
        reconstructed = array(geometry(mano, initial[0], initial[1]))
    expected = np.concatenate([source[f"{i}_nominal_pred_keypoints_3d"] for i in range(n)])
    diff = float(np.max(abs(reconstructed - expected)))
    assert diff < 1e-6, ("MANO_origin_or_rotation_mismatch", diff)
    K, R, ext_t = tensor(raw["K"][:, None]), tensor(np.eye(3)[None]), tensor(np.zeros((1, 3)))
    obs = tensor(np.stack(pseudo)[:, None])
    conf = torch.ones((n, 1, 21), device="cuda")
    gt, gt_uv = raw["xyz"], raw["uv_vis"][:, None, :, :2]
    visible, scale = target["visible"][:, None], target["affine"][:, 0, 0, None]
    np.savez_compressed(out / "inputs.npz", ids=ids, rotations=array(initial[0]), betas=array(initial[1]),
                        translation=array(initial[2]), pseudo_uv=array(obs), K=array(K), R=array(R),
                        ext_t=array(ext_t), confidence=array(conf), gt=gt, gt_uv=gt_uv,
                        visible=visible, affine_scale=scale)
    base = baseline(mano, initial, K, R, ext_t)
    np.savez_compressed(out / "initial.npz", **base)
    results = {"initial": metrics(base, gt, gt_uv, visible, scale, initial)}
    for mode in ("translation", "translation_orientation", "translation_pose", "translation_pose_shape"):
        result = solve(mano, out, mode, initial, obs, K, R, ext_t, conf, mode)
        results[mode] = metrics(result, gt, gt_uv, visible, scale, initial)
        (out / "results.partial.json").write_text(json.dumps(results, indent=2))
    return {"experiment": "E1", "sample_ids": ids.tolist(), "mano_reconstruction_max_difference_m": diff,
            "results": results, "input_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (teacher_dir / "outputs.npz", teacher_dir / "results.json", target_path, raw_path)},
            "limitations": "32 synthetic RHD development images; GT crops; teacher training overlap unknown; no stereo; GT is evaluation-only."}


def synthetic(mano, out, cfg):
    n = 8
    # Canonical valid columns avoid ambiguity in the legacy mean-pose asset.
    mean = tensor([1, 0, 0, 0, 1, 0]).reshape(1, 1, 6).expand(n, 16, -1)
    truth6 = mean + torch.randn_like(mean) * .1
    truth_rot = rot6d_to_rotmat(truth6).reshape(n, 16, 3, 3)
    beta = torch.randn(n, 10, device="cuda") * .3
    truth_t = tensor([[0, 0, .5 + .04*i] for i in range(n)])
    initial = (rot6d_to_rotmat(truth6 + torch.randn_like(truth6)*.03).reshape(n, 16, 3, 3),
               beta.clone(), truth_t + tensor([.015, -.01, .06]))
    angle = .08
    right_R = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                        [-np.sin(angle), 0, np.cos(angle)]])
    R = tensor(np.stack([np.eye(3), right_R]))
    ext_t = tensor(np.stack([np.zeros(3), -right_R @ np.array([.06, 0, 0])]))
    K = tensor(np.tile(np.array([[800, 0, 320], [0, 780, 240], [0, 0, 1]]), (n, 2, 1, 1)))
    with torch.no_grad():
        gt = geometry(mano, truth_rot, beta) + truth_t[:, None]
        clean, _ = project(gt, K, R, ext_t)
    noisy = clean + torch.randn_like(clean)
    noisy[:, 1, [4, 8, 12, 16]] += tensor([50, -35])
    confidence = torch.ones(n, 2, 21, device="cuda")
    confidence[:, 1, [4, 8, 12, 16]] = 0
    np.savez_compressed(out / "inputs.npz", gt=array(gt), clean=array(clean), noisy=array(noisy),
                        confidence=array(confidence), K=array(K), R=array(R), ext_t=array(ext_t),
                        rotations=array(initial[0]), betas=array(initial[1]), translation=array(initial[2]))
    results = {}
    base = baseline(mano, initial, K, R, ext_t)
    np.savez_compressed(out / "initial.npz", **base)
    results["initial"] = metrics(base, array(gt), array(clean), np.ones((n, 2, 21)), np.ones((n, 2)), initial)
    for name, views, robust, weighted, noiseless in [
        ("left", [0], False, False, False), ("right", [1], False, False, False),
        ("stereo", [0, 1], False, False, False), ("stereo_robust", [0, 1], True, False, False),
        ("stereo_confidence", [0, 1], False, True, False),
        ("stereo_clean_control", [0, 1], False, False, True)]:
        observations = (clean if noiseless else noisy)[:, views]
        conf = confidence[:, views] if weighted else torch.ones_like(confidence[:, views])
        result = solve(mano, out, name, initial, observations, K[:, views], R[views], ext_t[views],
                       conf, "translation_pose", robust)
        results[name] = metrics(result, array(gt), array(clean[:, views]),
                                np.ones((n, len(views), 21)), np.ones((n, len(views))), initial)
        (out / "results.partial.json").write_text(json.dumps(results, indent=2))
    return {"experiment": "E2_synthetic_control_only", "results": results,
            "limitations": "Generated MANO points, no stereo images or frontend predictions; known fixed betas; oracle outlier confidence. Not real E2."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["e1", "synthetic"], required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    out = ROOT / "artifacts" / args.run_id
    out.mkdir(exist_ok=False)
    for path in (Path(__file__), ROOT / "docs/dyn_hamr_reproduction_protocol.md"):
        shutil.copy2(path, out)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = CfgNode.load_cfg((ROOT / "inputs/model_config.yaml").read_text())
    cfg.MANO.DATA_DIR = str(ROOT / "inputs/data")
    cfg.MANO.MODEL_PATH = str(ROOT / "inputs/data/mano")
    cfg.MANO.MEAN_PARAMS = str(ROOT / "inputs/data/mano_mean_params.npz")
    mano = MANO(**{k.lower(): v for k, v in dict(cfg.MANO).items()}).cuda().eval()
    mano.requires_grad_(False)
    (out / "effective_config.json").write_text(json.dumps({
        "experiment": args.experiment, "steps": 800, "seed": 42, "pose_prior_weight": .1,
        "shape_prior_weight": .01, "reprojection_scale_px": 3, "translation_lr": .001,
        "rotation_beta_lr": .01, "lr_decay_step": 600, "deterministic": True,
        "device": torch.cuda.get_device_name(), "torch": torch.__version__}, indent=2))
    begin = time.perf_counter()
    report = e1(mano, out) if args.experiment == "e1" else synthetic(mano, out, cfg)
    report["elapsed_s"] = time.perf_counter() - begin
    report["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated()
    (out / "results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"completed": args.run_id, "elapsed_s": report["elapsed_s"]}), flush=True)


if __name__ == "__main__":
    main()
