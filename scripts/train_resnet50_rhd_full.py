"""R1/R2 Fast-HaMeR-style ResNet-50 training on the complete usable RHD set."""
import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from yacs.config import CfgNode

ROOT = Path(__file__).resolve().parents[1]
for _name, _value in {"bool": np.bool_, "int": int, "float": float, "complex": complex,
                      "object": object, "unicode": str, "str": str}.items():
    if _name not in np.__dict__:
        setattr(np, _name, _value)
sys.path.insert(0, str(ROOT / "repos/Fast-HaMeR"))
from hamer.models.hamer import HAMER

TIPS = [4, 8, 12, 16, 20]
FINGERS = {"thumb": [0, 1, 2, 3, 4], "index": [0, 5, 6, 7, 8],
           "middle": [0, 9, 10, 11, 12], "ring": [0, 13, 14, 15, 16],
           "little": [0, 17, 18, 19, 20]}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def positive_camera_hook(module, inputs, output):
    params, camera, history = output
    camera = torch.cat((torch.nn.functional.softplus(camera[:, :1]) + .05,
                        camera[:, 1:]), dim=1)
    history = dict(history)
    if "cam" in history:
        raw = history["cam"]
        history["cam"] = torch.cat((torch.nn.functional.softplus(raw[:, :1]) + .05,
                                    raw[:, 1:]), dim=1)
    return params, camera, history


def configure_model(config):
    cfg = CfgNode.load_cfg((ROOT / "inputs/model_config.yaml").read_text())
    official = yaml.safe_load(
        (ROOT / "repos/Fast-HaMeR/hamer/configs_hydra/experiment/resnet_50.yaml").read_text()
    )
    cfg.MODEL.BACKBONE = CfgNode(official["MODEL"]["BACKBONE"])
    cfg.MODEL.MANO_HEAD = CfgNode(official["MODEL"]["MANO_HEAD"])
    cfg.MODEL.IMAGE_SIZE = config["image_size"]
    cfg.MANO.DATA_DIR = str(ROOT / "inputs/data")
    cfg.MANO.MODEL_PATH = str(ROOT / "inputs/data/mano")
    cfg.MANO.MEAN_PARAMS = str(ROOT / "inputs/data/mano_mean_params.npz")
    cfg.EXTRA.FOCAL_LENGTH = 5000.
    for name in ("KEYPOINTS_3D", "KEYPOINTS_2D", "GLOBAL_ORIENT",
                 "HAND_POSE", "BETAS", "ADVERSARIAL"):
        cfg.LOSS_WEIGHTS[name] = 0.
    torch.hub.set_dir(str(ROOT / "inputs/torch_hub"))
    weight = ROOT / "inputs/resnet50-11ad3fa6.pth"
    assert sha(weight).startswith("11ad3fa6")
    cache = ROOT / "inputs/torch_hub/checkpoints" / weight.name
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        os.link(weight, cache)
    model = HAMER(cfg, init_renderer=False)
    initialization = config["head_initialization"]
    for name in ("decpose", "decshape", "deccam"):
        layer = getattr(model.mano_head, name)
        torch.nn.init.xavier_uniform_(layer.weight, gain=initialization["gain"])
        torch.nn.init.constant_(layer.bias, initialization["bias"])
    target_scale = config["positive_camera"]["initial_output_scale"]
    raw_scale = target_scale - config["positive_camera"]["minimum_scale"]
    raw_scale = raw_scale + math.log(-math.expm1(-raw_scale))
    with torch.no_grad():
        model.mano_head.init_cam.copy_(
            torch.tensor([[raw_scale, 0., 0.]], dtype=model.mano_head.init_cam.dtype))
    model.mano_head.register_forward_hook(positive_camera_hook)
    return model, cfg, weight


class RHDDataset(Dataset):
    def __init__(self, labels, train, config):
        raw = np.load(labels, allow_pickle=False)
        self.data = {key: raw[key] for key in raw.files}
        self.train = train
        self.config = config
        self.image_size = config["image_size"]
        self.root = ROOT / "inputs/rhd"

    def __len__(self):
        return len(self.data["sample_id"])

    def __getitem__(self, index):
        image = cv2.imread(str(self.root / str(self.data["image"][index])))
        if image is None:
            raise FileNotFoundError(str(self.data["image"][index]))
        uv = self.data["uv_vis"][index, :, :2].copy()
        visible = self.data["uv_vis"][index, :, 2] > 0
        xyz = self.data["xyz_absolute_m"][index].copy()
        xyz -= xyz[:1]
        right = bool(self.data["right"][index])
        fixed = self.data["affine_full_to_crop"][index]
        base_scale = float(fixed[0, 0])
        base_center = np.asarray([(self.image_size / 2 - fixed[0, 2]) / base_scale,
                                  (self.image_size / 2 - fixed[1, 2]) / base_scale])
        side = self.image_size / base_scale
        angle = 0.
        color = np.ones(3)
        if self.train:
            augment = self.config["augmentation"]
            rng = np.random
            scale_noise = np.clip(rng.randn(), -1., 1.) * augment["scale_factor"]
            side *= 1 + scale_noise
            if rng.rand() < augment["rotation_rate"]:
                angle = float(np.clip(rng.randn() * augment["rotation_factor_deg"],
                                      -2 * augment["rotation_factor_deg"],
                                      2 * augment["rotation_factor_deg"]))
            if rng.rand() < augment["translation_rate"]:
                base_center += rng.uniform(-1, 1, 2) * augment["translation_factor"] * side
            color = rng.uniform(1 - augment["color_scale"], 1 + augment["color_scale"], 3)
        if not np.isfinite(side) or side <= 0:
            raise ValueError(f"invalid crop side {side} for sample {self.data['sample_id'][index]}")
        theta = np.radians(angle)
        rotation = np.asarray([[np.cos(theta), np.sin(theta)],
                               [-np.sin(theta), np.cos(theta)]], np.float32)
        affine = np.zeros((2, 3), np.float32)
        affine[:, :2] = rotation * (self.image_size / side)
        affine[:, 2] = self.image_size / 2 - affine[:, :2] @ base_center
        crop = cv2.warpAffine(image, affine, (self.image_size, self.image_size),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        crop_uv = np.c_[uv, np.ones(21)] @ affine.T
        xyz[:, :2] = xyz[:, :2] @ rotation.T
        if not right:
            crop = crop[:, ::-1].copy()
            crop_uv[:, 0] = self.image_size - 1 - crop_uv[:, 0]
            xyz[:, 0] *= -1
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
        crop = np.clip(crop * color[None, None], 0, 255)
        crop = (crop / 255 - np.asarray([.485, .456, .406])) / np.asarray([.229, .224, .225])
        return {
            "img": torch.from_numpy(crop.transpose(2, 0, 1).astype(np.float32)),
            "xyz": torch.from_numpy(xyz.astype(np.float32)),
            "uv": torch.from_numpy((crop_uv / self.image_size - .5).astype(np.float32)),
            "visible": torch.from_numpy(visible),
            "sample_id": str(self.data["sample_id"][index]),
            "index": int(index),
        }


def worker_init(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def camera_depth(output):
    return output["pred_keypoints_3d"][:, :, 2] + output["pred_cam_t"][:, None, 2]


def losses(output, batch, weights):
    relative = output["pred_keypoints_3d"] - output["pred_keypoints_3d"][:, :1]
    visible = batch["visible"]
    terms = {
        "joints3d_wrist_relative_l1": (relative - batch["xyz"]).abs().mean(),
        "visible_joints2d_l1": (output["pred_keypoints_2d"] - batch["uv"]).abs()[visible].mean(),
    }
    total = sum(terms[name] * weights[name] for name in terms)
    return total, terms


def kabsch_errors(pred, target):
    result = []
    for p, t in zip(pred, target):
        p, t = p[1:] - p[:1], t[1:] - t[:1]
        u, _, vt = np.linalg.svd(p.T @ t)
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            vt[-1] *= -1
            rotation = vt.T @ u.T
        result.append(np.linalg.norm(p @ rotation.T - t, axis=1).mean() * 1000)
    return np.asarray(result)


def evaluate(model, loader, device, out, split, step, save_overlay=True):
    model.eval()
    xyzs, targets, uvs, target_uvs, visibles, cameras, depths, indices = ([] for _ in range(8))
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for batch in loader:
            images = batch["img"].to(device, non_blocking=True)
            output = model.forward_step({"img": images})
            xyzs.append(output["pred_keypoints_3d"].cpu().numpy())
            targets.append(batch["xyz"].numpy())
            uvs.append(output["pred_keypoints_2d"].cpu().numpy())
            target_uvs.append(batch["uv"].numpy())
            visibles.append(batch["visible"].numpy())
            cameras.append(output["pred_cam"].cpu().numpy())
            depths.append(camera_depth(output).cpu().numpy())
            indices.extend(batch["index"].numpy().tolist())
    xyz = np.concatenate(xyzs)
    xyz -= xyz[:, :1]
    target = np.concatenate(targets)
    uv, target_uv = np.concatenate(uvs), np.concatenate(target_uvs)
    visible = np.concatenate(visibles)
    camera, depth = np.concatenate(cameras), np.concatenate(depths)
    finite = (np.isfinite(xyz).all((1, 2)) & np.isfinite(uv).all((1, 2)) &
              np.isfinite(camera).all(1))
    positive = (depth > 0).all(1)
    span = np.ptp(uv, axis=1).max(1) * 256
    collapse = span < 5
    errors3d = np.linalg.norm(xyz - target, axis=-1) * 1000
    errors2d = np.linalg.norm(uv - target_uv, axis=-1) * 256
    tips = errors3d[:, TIPS]
    bone = {}
    for name, chain in FINGERS.items():
        p, t = np.diff(xyz[:, chain], axis=1), np.diff(target[:, chain], axis=1)
        cosine = np.sum(p * t, axis=-1) / np.maximum(
            np.linalg.norm(p, axis=-1) * np.linalg.norm(t, axis=-1), 1e-12)
        bone[name] = float(np.degrees(np.arccos(np.clip(cosine, -1, 1))).mean())
    metrics = {
        "step": step,
        "split": split,
        "count": int(len(xyz)),
        "wrist_relative_mpjpe_mm": float(errors3d.mean()),
        "rotation_aligned_diagnostic_mpjpe_mm": float(kabsch_errors(xyz, target).mean()),
        "visible_2d_epe_px": float(errors2d[visible].mean()),
        "fingertip_wrist_relative_error_mm": float(tips.mean()),
        "per_finger_bone_direction_error_deg": bone,
        "projection_bbox_span_px": {
            "min": float(span.min()), "median": float(np.median(span)),
            "p95": float(np.percentile(span, 95)), "max": float(span.max()),
        },
        "weak_camera_scale": {
            "min": float(camera[:, 0].min()), "median": float(np.median(camera[:, 0])),
            "max": float(camera[:, 0].max()),
        },
        "nonfinite_rate": float((~finite).mean()),
        "negative_depth_rate": float((~positive).mean()),
        "projection_collapse_rate": float(collapse.mean()),
        "failure_rate": float((~finite | ~positive | collapse).mean()),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    prediction_path = out / f"{split}_predictions_step_{step:07d}.npz"
    np.savez_compressed(prediction_path, index=np.asarray(indices), xyz_wrist_relative_m=xyz,
                        uv_normalized=uv, weak_camera=camera, depth_m=depth,
                        error3d_mm=errors3d, error2d_px=errors2d, visible=visible)
    if save_overlay:
        dataset = loader.dataset
        tiles = []
        for position in range(min(8, len(indices))):
            sample = dataset[indices[position]]
            rgb = sample["img"].numpy().transpose(1, 2, 0)
            rgb = np.clip((rgb * np.asarray([.229, .224, .225]) +
                           np.asarray([.485, .456, .406])) * 255, 0, 255).astype(np.uint8)
            for points, color in ((target_uv[position], (0, 255, 0)),
                                  (uv[position], (255, 0, 0))):
                for x, y in (points + .5) * 256:
                    cv2.circle(rgb, (round(float(x)), round(float(y))), 2, color, -1)
            tiles.append(rgb)
        contact = np.concatenate(tiles, axis=1)
        cv2.imwrite(str(out / f"{split}_overlay_step_{step:07d}.jpg"),
                    cv2.cvtColor(contact, cv2.COLOR_RGB2BGR))
    with (out / "evaluations.jsonl").open("a") as handle:
        handle.write(json.dumps(metrics, allow_nan=False) + "\n")
    print(json.dumps({"evaluation": metrics}, allow_nan=False), flush=True)
    return metrics


def benchmark(model, sample, device):
    model.eval()
    image = sample["img"][None].to(device)
    with torch.no_grad():
        for _ in range(20):
            model.forward_step({"img": image})
        torch.cuda.synchronize(device)
        times = []
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(100):
            start = time.perf_counter()
            model.forward_step({"img": image})
            torch.cuda.synchronize(device)
            times.append((time.perf_counter() - start) * 1000)
    return {
        "protocol": "batch=1, 20 warmup, 100 timed, CUDA synchronize, network+MANO only",
        "mean_ms": float(np.mean(times)), "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def save_checkpoint(path, model, optimizer, scheduler, step, epoch, config, best):
    torch.save({
        "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "step": step, "epoch": epoch, "config": config, "best": best,
        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
        "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
    }, path)


def strict_reload(path, cfg, config, device, sample):
    checkpoint = torch.load(path, map_location="cpu")
    model = HAMER(cfg, init_renderer=False)
    model.mano_head.register_forward_hook(positive_camera_hook)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    with torch.no_grad():
        output = model.forward_step({"img": sample["img"][None].to(device)})
    return model, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/resnet50_rhd_no_kd_v1.json")
    parser.add_argument("--stage", choices=("smoke", "formal"), required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text())
    assert config["teacher"] is None and not config["kd"] and not config["adversarial"]
    out = ROOT / "artifacts" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    dataset_root = ROOT / config["dataset"]
    manifest = json.loads((dataset_root / "manifest.json").read_text())
    assert manifest["real_e2_exclusion"].startswith("No DexYCB")
    shutil.copy2(config_path, out / config_path.name)
    shutil.copy2(dataset_root / "manifest.json", out / "rhd_manifest.json")
    shutil.copy2(Path(__file__), out / Path(__file__).name)
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    train_data = RHDDataset(dataset_root / "train.npz", True, config)
    dev_data = RHDDataset(dataset_root / "dev.npz", False, config)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True,
                              num_workers=config["workers"], pin_memory=True, drop_last=True,
                              persistent_workers=True, worker_init_fn=worker_init,
                              generator=generator)
    dev_loader = DataLoader(dev_data, batch_size=config["batch_size"], shuffle=False,
                            num_workers=config["workers"], pin_memory=True,
                            persistent_workers=True, worker_init_fn=worker_init)
    model, cfg, imagenet_weight = configure_model(config)
    model.to(device)
    optimizer = torch.optim.AdamW(model.get_parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])
    updates_per_epoch = len(train_loader)
    if args.stage == "smoke":
        maximum_updates = config["smoke_updates"]
        eval_steps = set(config["smoke_snapshots"])
        scheduler = None
    else:
        maximum_updates = updates_per_epoch * config["formal_epochs"]
        eval_steps = set(range(updates_per_epoch, maximum_updates + 1, updates_per_epoch))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(maximum_updates, 1), eta_min=config["lr"] * .05)
    step, start_epoch, best = 0, 0, None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = checkpoint["step"]
        start_epoch = checkpoint["epoch"]
        if scheduler and checkpoint.get("scheduler"):
            scheduler.load_state_dict(checkpoint["scheduler"])
        best = checkpoint.get("best")
    initial_backbone = next(model.backbone.parameters()).detach().cpu().clone()
    initial_head = model.mano_head.decpose.weight.detach().cpu().clone()
    (out / "effective_config.yaml").write_text(cfg.dump())
    provenance = {
        "stage": args.stage, "run_id": args.run_id,
        "start_utc": datetime.now(timezone.utc).isoformat(),
        "model_initialization": "ImageNet1K V2 backbone plus randomly initialized Fast-HaMeR MANO head; no hand checkpoint",
        "imagenet_weight_sha256": sha(imagenet_weight),
        "resume": str(args.resume) if args.resume else None,
        "resume_sha256": sha(args.resume) if args.resume else None,
        "train_instances": len(train_data), "dev_instances": len(dev_data),
        "updates_per_epoch": updates_per_epoch, "maximum_updates": maximum_updates,
        "loss_weights": config["loss_weights"],
        "forbidden_final_e3_use": True,
    }
    write_json(out / "provenance.json", provenance)
    initial = evaluate(model, dev_loader, device, out, "dev", step)
    initial_score = (initial["wrist_relative_mpjpe_mm"], initial["visible_2d_epe_px"],
                     initial["fingertip_wrist_relative_error_mm"], initial["failure_rate"])
    best = {"step": step, "epoch": start_epoch, "score": list(initial_score),
            "metrics": initial}
    save_checkpoint(out / "best_dev.pt", model, optimizer, scheduler,
                    step, start_epoch, config, best)
    if args.stage == "smoke":
        save_checkpoint(out / "step_0000000.pt", model, optimizer, scheduler,
                        step, start_epoch, config, best)
    snapshots = {step}
    bad_evals = 0
    start_time = time.perf_counter()
    epoch = start_epoch
    model.train()
    while step < maximum_updates:
        epoch += 1
        for batch in train_loader:
            if step >= maximum_updates:
                break
            model.train()
            batch = {key: value.to(device, non_blocking=True)
                     if torch.is_tensor(value) else value for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model.forward_step({"img": batch["img"]}, train=True)
            total, terms = losses(output, batch, config["loss_weights"])
            depth = camera_depth(output)
            span = torch.max(torch.max(output["pred_keypoints_2d"], dim=1).values -
                             torch.min(output["pred_keypoints_2d"], dim=1).values, dim=1).values * 256
            if (not torch.isfinite(total) or not torch.isfinite(depth).all() or
                    not torch.isfinite(output["pred_keypoints_2d"]).all()):
                raise FloatingPointError(f"nonfinite at step {step + 1}")
            component_gradients = None
            if step == 0 or (step + 1) % 100 == 0:
                targets = (model.mano_head.decpose.weight, next(model.backbone.parameters()))
                component_gradients = {}
                for name, term in terms.items():
                    grads = torch.autograd.grad(
                        term * config["loss_weights"][name], targets,
                        retain_graph=True, allow_unused=True)
                    component_gradients[name] = {
                        "head_decpose": None if grads[0] is None else float(grads[0].norm()),
                        "backbone_first": None if grads[1] is None else float(grads[1].norm()),
                    }
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.get_parameters(), config["clip_grad_norm"], error_if_nonfinite=True)
            optimizer.step()
            if scheduler:
                scheduler.step()
            step += 1
            row = {
                "step": step, "epoch": epoch, "loss": float(total.detach()),
                "loss_components": {name: float(value.detach()) for name, value in terms.items()},
                "component_weighted_gradient_norms": component_gradients,
                "total_gradient_norm_before_clip": float(grad_norm),
                "lr": optimizer.param_groups[0]["lr"],
                "weak_camera_scale_min_max": [
                    float(output["pred_cam"][:, 0].min()), float(output["pred_cam"][:, 0].max())],
                "camera_depth_min_max_m": [float(depth.min()), float(depth.max())],
                "projection_span_px_min_median_max": [
                    float(span.min()), float(span.median()), float(span.max())],
                "elapsed_s": time.perf_counter() - start_time,
            }
            with (out / "steps.jsonl").open("a") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
            if step <= 5 or step % 25 == 0:
                print(json.dumps(row, allow_nan=False), flush=True)
            if step in eval_steps:
                metrics = evaluate(model, dev_loader, device, out, "dev", step)
                score = (metrics["wrist_relative_mpjpe_mm"], metrics["visible_2d_epe_px"],
                         metrics["fingertip_wrist_relative_error_mm"], metrics["failure_rate"])
                improved = best is None or score < tuple(best["score"])
                if improved:
                    best = {"step": step, "epoch": epoch, "score": list(score), "metrics": metrics}
                    save_checkpoint(out / "best_dev.pt", model, optimizer, scheduler,
                                    step, epoch, config, best)
                    bad_evals = 0
                else:
                    bad_evals += 1
                if args.stage == "smoke" or step in config["smoke_snapshots"]:
                    save_checkpoint(out / f"step_{step:07d}.pt", model, optimizer,
                                    scheduler, step, epoch, config, best)
                if (args.stage == "formal" and epoch >= config["formal_min_epochs"] and
                        bad_evals >= config["early_stopping_patience_evals"]):
                    maximum_updates = step
                    break
    save_checkpoint(out / "last.pt", model, optimizer, scheduler, step, epoch, config, best)
    model.eval()
    sample = dev_data[0]
    with torch.no_grad():
        before = model.forward_step({"img": sample["img"][None].to(device)})
    del optimizer
    restored, after = strict_reload(out / "last.pt", cfg, config, device, sample)
    reload_xyz = float((before["pred_keypoints_3d"] - after["pred_keypoints_3d"]).abs().max())
    reload_uv = float((before["pred_keypoints_2d"] - after["pred_keypoints_2d"]).abs().max())
    assert reload_xyz == 0 and reload_uv == 0
    backbone_update = float((next(restored.backbone.parameters()).detach().cpu() -
                             initial_backbone).abs().max())
    head_update = float((restored.mano_head.decpose.weight.detach().cpu() -
                         initial_head).abs().max())
    final = evaluate(restored, dev_loader, device, out, "dev_reload", step)
    latency = benchmark(restored, sample, device)
    smoke_passed = (
        np.isfinite(final["wrist_relative_mpjpe_mm"]) and
        final["weak_camera_scale"]["min"] > 0 and
        final["nonfinite_rate"] == 0 and final["negative_depth_rate"] == 0 and
        final["projection_bbox_span_px"]["median"] > 20 and
        final["projection_collapse_rate"] == 0 and
        backbone_update > 0 and head_update > 0 and reload_xyz == 0 and reload_uv == 0
    )
    if args.stage == "smoke":
        assert smoke_passed
    result = {
        "stage": args.stage, "passed": bool(smoke_passed if args.stage == "smoke" else True),
        "actual_updates": step, "actual_epochs": epoch, "initial": initial, "final": final,
        "best_dev": best, "strict_reload_xyz_max_m": reload_xyz,
        "strict_reload_uv_max_normalized": reload_uv,
        "backbone_first_parameter_max_update": backbone_update,
        "head_decpose_max_update": head_update, "latency": latency,
        "last_checkpoint_sha256": sha(out / "last.pt"),
        "best_checkpoint_sha256": sha(out / "best_dev.pt") if (out / "best_dev.pt").exists() else None,
        "end_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(out / "results.json", result)
    print(json.dumps({"completed": result}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
