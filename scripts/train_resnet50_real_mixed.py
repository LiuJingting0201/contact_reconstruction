"""Train the no-KD real-mixed ResNet-50, starting with the M-S0 subset smoke."""
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
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from yacs.config import CfgNode

ROOT = Path(__file__).resolve().parents[1]
for _name, _value in {
    "bool": np.bool_, "int": int, "float": float, "complex": complex,
    "object": object, "unicode": str, "str": str,
}.items():
    if _name not in np.__dict__:
        setattr(np, _name, _value)
sys.path.insert(0, str(ROOT / "repos/Fast-HaMeR"))
from hamer.models.hamer import HAMER


EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
TIPS = [4, 8, 12, 16, 20]
MEAN = np.asarray([0.485, 0.456, 0.406], np.float32)
STD = np.asarray([0.229, 0.224, 0.225], np.float32)
DATASET_IDS = {"dexycb": 0, "freihand": 1, "rhd": 2}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def normalize(image_rgb, color=None):
    image = image_rgb.astype(np.float32)
    if color is not None:
        image = np.clip(image * color[None, None], 0, 255)
    return ((image / 255.0 - MEAN) / STD).transpose(2, 0, 1).astype(np.float32)


def axis_angle_to_matrix(axis_angle):
    values = np.asarray(axis_angle, np.float32).reshape(-1, 3)
    return np.stack([cv2.Rodrigues(value)[0] for value in values]).astype(np.float32)


def mirror_rotations(rotations):
    reflection = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)
    return reflection[None] @ rotations @ reflection[None]


def empty_targets():
    return {
        "global_orient": np.eye(3, dtype=np.float32),
        "global_orient_valid": False,
        "hand_pose": np.tile(np.eye(3, dtype=np.float32), (15, 1, 1)),
        "hand_pose_valid": False,
        "betas": np.zeros(10, np.float32),
        "betas_valid": False,
    }


def package_sample(image, xyz, uv, uv_valid, sample_id, dataset, targets):
    xyz_valid = bool(np.isfinite(xyz).all())
    uv_valid = np.asarray(uv_valid, bool) & np.isfinite(uv).all(1)
    # Missing/invalid targets are numerically sanitized only behind false masks.
    # They are never used as zero-valued supervision.
    targets = dict(targets)
    for key in ("global_orient", "hand_pose", "betas"):
        targets[key + "_valid"] = bool(
            targets[key + "_valid"] and np.isfinite(targets[key]).all())
        if not targets[key + "_valid"]:
            targets[key] = empty_targets()[key]
    value = {
        "img": torch.from_numpy(normalize(image)),
        "xyz": torch.from_numpy(np.nan_to_num(np.asarray(xyz, np.float32))),
        "xyz_valid": torch.tensor(xyz_valid),
        "uv": torch.from_numpy(np.nan_to_num(np.asarray(uv, np.float32))),
        "uv_valid": torch.from_numpy(np.asarray(uv_valid, bool)),
        "global_orient": torch.from_numpy(targets["global_orient"]),
        "global_orient_valid": torch.tensor(targets["global_orient_valid"]),
        "hand_pose": torch.from_numpy(targets["hand_pose"]),
        "hand_pose_valid": torch.tensor(targets["hand_pose_valid"]),
        "betas": torch.from_numpy(targets["betas"]),
        "betas_valid": torch.tensor(targets["betas_valid"]),
        "dataset_id": torch.tensor(DATASET_IDS[dataset], dtype=torch.int64),
        "sample_id": sample_id,
    }
    return value


class DexYCBDataset(Dataset):
    def __init__(self, root, image_size, augmentation, train):
        self.root = root
        self.image_size = image_size
        self.augmentation = augmentation
        self.train = train
        self.manifest = json.loads((root / "sealed_manifest.json").read_text())
        if self.manifest["split"] != "train":
            raise ValueError("M-S0 DexYCB input must be a train split")
        if self.manifest["subject"] != "20200813-subject-02":
            raise ValueError("M-S0 is preregistered for subject-02 only")
        self.rows = []
        for sample in self.manifest["samples"]:
            for view in sample["views"]:
                self.rows.append({
                    "sample_id": sample["sample_id"],
                    "mano_side": sample["mano_side"],
                    **view,
                })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = cv2.imread(str(self.root / row["image"]))
        labels = np.load(self.root / row["label"])
        if image is None:
            raise FileNotFoundError(self.root / row["image"])
        uv = labels["joint_2d"][0].astype(np.float32)
        xyz_absolute = labels["joint_3d"][0].astype(np.float32)
        source_valid = (
            np.isfinite(uv).all(1)
            & np.isfinite(xyz_absolute).all(1)
            & (uv >= 0).all(1)
        )
        if source_valid.sum() < 4 or not (xyz_absolute[:, 2] > 0).all():
            raise ValueError(f"invalid DexYCB labels: {row['label']}")
        lo, hi = uv[source_valid].min(0), uv[source_valid].max(0)
        center = (lo + hi) / 2
        side = max(float((hi - lo).max()) * self.augmentation["dexycb_crop_rescale"], 64.0)
        color = np.ones(3, np.float32)
        if self.train:
            side *= 1.0 + np.random.uniform(
                -self.augmentation["dexycb_scale_factor"],
                self.augmentation["dexycb_scale_factor"],
            )
            center += np.random.uniform(-1, 1, 2) * (
                self.augmentation["dexycb_translation_factor"] * side
            )
            factor = self.augmentation["color_scale"]
            color = np.random.uniform(1 - factor, 1 + factor, 3).astype(np.float32)
        scale = self.image_size / side
        affine = np.asarray([
            [scale, 0.0, self.image_size / 2 - scale * center[0]],
            [0.0, scale, self.image_size / 2 - scale * center[1]],
        ], np.float32)
        crop = cv2.warpAffine(
            image, affine, (self.image_size, self.image_size),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        )
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = np.clip(crop.astype(np.float32) * color[None, None], 0, 255).astype(np.uint8)
        crop_uv = np.c_[uv, np.ones(21, np.float32)] @ affine.T
        xyz = xyz_absolute - xyz_absolute[:1]
        targets = empty_targets()
        pose_m = labels["pose_m"][0]
        if np.isfinite(pose_m).all() and np.any(np.abs(pose_m) > 1e-8):
            targets["global_orient"] = axis_angle_to_matrix(pose_m[:3])[0]
            targets["global_orient_valid"] = True
        right = row["mano_side"] == "right"
        if not right:
            crop = crop[:, ::-1].copy()
            crop_uv[:, 0] = self.image_size - 1 - crop_uv[:, 0]
            xyz[:, 0] *= -1
            targets["global_orient"] = mirror_rotations(
                targets["global_orient"][None]
            )[0]
        uv_valid = (
            source_valid
            & (crop_uv >= 0).all(1)
            & (crop_uv < self.image_size).all(1)
        )
        sample = package_sample(
            crop, xyz, crop_uv / self.image_size - 0.5, uv_valid,
            f"{row['sample_id']}:{row['camera_serial']}", "dexycb", targets,
        )
        return sample


class FreiHANDDataset(Dataset):
    def __init__(self, root, image_size, augmentation, train, versions=1):
        self.root = root
        self.versions = versions
        self.image_size = image_size
        self.augmentation = augmentation
        self.train = train
        self.K = np.asarray(json.loads((root / "training_K.json").read_text()), np.float32)
        self.xyz = np.asarray(
            json.loads((root / "training_xyz.json").read_text()), np.float32
        )
        self.mano = np.asarray(
            json.loads((root / "training_mano.json").read_text()), np.float32
        )[:, 0]
        expected = (32560, 21, 3)
        if self.xyz.shape != expected or self.mano.shape != (32560, 61):
            raise ValueError("unexpected FreiHAND annotation shape")

    def __len__(self):
        return len(self.xyz) * self.versions

    def __getitem__(self, index):
        path = self.root / "training" / "rgb" / f"{index:08d}.jpg"
        image_index = index
        index %= len(self.xyz)
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.train and np.random.rand() < self.augmentation[
                "freihand_background_replace_rate"]:
            mask_path = self.root / "training" / "mask" / f"{index:08d}.jpg"
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(mask_path)
            foreground = (mask.astype(np.float32) / 255.0)[..., None]
            base = np.random.uniform(20, 235, (1, 1, 3))
            noise = np.random.normal(0, 8, image.shape)
            background = np.clip(base + noise, 0, 255)
            image = np.clip(
                image * foreground + background * (1 - foreground), 0, 255
            ).astype(np.uint8)
        factor = self.augmentation["color_scale"] if self.train else 0.0
        color = np.random.uniform(1 - factor, 1 + factor, 3) if factor else np.ones(3)
        image = np.clip(image.astype(np.float32) * color[None, None], 0, 255).astype(
            np.uint8
        )
        image = cv2.resize(
            image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
        )
        homogeneous = self.xyz[index] @ self.K[index].T
        uv = homogeneous[:, :2] / homogeneous[:, 2:]
        uv *= self.image_size / 224.0
        uv_valid = (
            np.isfinite(uv).all(1)
            & (uv >= 0).all(1)
            & (uv < self.image_size).all(1)
        )
        xyz = self.xyz[index] - self.xyz[index, :1]
        targets = empty_targets()
        rotations = axis_angle_to_matrix(self.mano[index, :48])
        targets.update({
            "global_orient": rotations[0],
            "global_orient_valid": True,
            "hand_pose": rotations[1:],
            "hand_pose_valid": True,
            "betas": self.mano[index, 48:58].copy(),
            "betas_valid": True,
        })
        return package_sample(
            image, xyz, uv / self.image_size - 0.5, uv_valid,
            f"freihand:{index:05d}:rgb{image_index:08d}", "freihand", targets,
        )


class RHDDataset(Dataset):
    def __init__(self, labels, image_root, image_size, augmentation, train):
        raw = np.load(labels, allow_pickle=False)
        self.data = {key: raw[key] for key in raw.files}
        self.image_root = image_root
        self.image_size = image_size
        self.augmentation = augmentation
        self.train = train

    def __len__(self):
        return len(self.data["sample_id"])

    def __getitem__(self, index):
        image_path = self.image_root / str(self.data["image"][index])
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        uv = self.data["uv_vis"][index, :, :2].copy()
        uv_valid = self.data["uv_vis"][index, :, 2] > 0
        xyz = self.data["xyz_absolute_m"][index].copy()
        xyz -= xyz[:1]
        affine = self.data["affine_full_to_crop"][index].copy()
        crop = cv2.warpAffine(
            image, affine, (self.image_size, self.image_size),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        )
        crop_uv = np.c_[uv, np.ones(21)] @ affine.T
        right = bool(self.data["right"][index])
        if not right:
            crop = crop[:, ::-1].copy()
            crop_uv[:, 0] = self.image_size - 1 - crop_uv[:, 0]
            xyz[:, 0] *= -1
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        uv_valid &= (
            np.isfinite(crop_uv).all(1)
            & (crop_uv >= 0).all(1)
            & (crop_uv < self.image_size).all(1)
        )
        if self.train:
            factor = self.augmentation["color_scale"]
            color = np.random.uniform(1 - factor, 1 + factor, 3)
            crop = np.clip(
                crop.astype(np.float32) * color[None, None], 0, 255
            ).astype(np.uint8)
        targets = empty_targets()
        return package_sample(
            crop, xyz, crop_uv / self.image_size - 0.5, uv_valid,
            str(self.data["sample_id"][index]), "rhd", targets,
        )


class FixedMixBatchSampler:
    def __init__(self, lengths, counts, updates, seed):
        self.lengths = lengths
        self.counts = counts
        self.updates = updates
        self.seed = seed

    def __len__(self):
        return self.updates

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        offsets = np.cumsum([0] + self.lengths[:-1]).tolist()
        permutations = [
            torch.randperm(length, generator=generator).tolist()
            for length in self.lengths
        ]
        positions = [0] * len(self.lengths)
        for _ in range(self.updates):
            batch = []
            for domain, (length, count, offset) in enumerate(
                    zip(self.lengths, self.counts, offsets)):
                for _ in range(count):
                    if positions[domain] == length:
                        permutations[domain] = torch.randperm(
                            length, generator=generator
                        ).tolist()
                        positions[domain] = 0
                    batch.append(offset + permutations[domain][positions[domain]])
                    positions[domain] += 1
            order = torch.randperm(len(batch), generator=generator).tolist()
            yield [batch[index] for index in order]


def worker_init(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def positive_camera_hook(module, inputs, output):
    params, camera, history = output
    camera = torch.cat(
        (torch.nn.functional.softplus(camera[:, :1]) + 0.05, camera[:, 1:]), dim=1
    )
    history = dict(history)
    if "cam" in history:
        raw = history["cam"]
        history["cam"] = torch.cat(
            (torch.nn.functional.softplus(raw[:, :1]) + 0.05, raw[:, 1:]), dim=1
        )
    return params, camera, history


def configure_model(config):
    cfg = CfgNode.load_cfg((ROOT / "inputs/model_config.yaml").read_text())
    official = yaml.safe_load(
        (ROOT / "repos/Fast-HaMeR/hamer/configs_hydra/experiment/resnet_50.yaml").read_text()
    )
    cfg.MODEL.BACKBONE = CfgNode(official["MODEL"]["BACKBONE"])
    cfg.MODEL.MANO_HEAD = CfgNode(official["MODEL"]["MANO_HEAD"])
    cfg.MODEL.IMAGE_SIZE = config["model"]["image_size"]
    cfg.MANO.DATA_DIR = str(ROOT / "inputs/data")
    cfg.MANO.MODEL_PATH = str(ROOT / "inputs/data/mano")
    cfg.MANO.MEAN_PARAMS = str(ROOT / "inputs/data/mano_mean_params.npz")
    cfg.EXTRA.FOCAL_LENGTH = 5000.0
    for name in (
        "KEYPOINTS_3D", "KEYPOINTS_2D", "GLOBAL_ORIENT",
        "HAND_POSE", "BETAS", "ADVERSARIAL",
    ):
        cfg.LOSS_WEIGHTS[name] = 0.0
    torch.hub.set_dir(str(ROOT / "inputs/torch_hub"))
    weight = ROOT / "inputs/resnet50-11ad3fa6.pth"
    if not sha256(weight).startswith("11ad3fa6"):
        raise ValueError("unexpected ImageNet1K V2 weight hash")
    cache = ROOT / "inputs/torch_hub/checkpoints" / weight.name
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        os.link(weight, cache)
    model = HAMER(cfg, init_renderer=False)
    for name in ("decpose", "decshape", "deccam"):
        layer = getattr(model.mano_head, name)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.01)
        torch.nn.init.constant_(layer.bias, 0.0)
    positive = config["positive_camera"]
    target_scale = positive["initial_output_scale"]
    raw_scale = target_scale - positive["minimum_scale"]
    raw_scale = raw_scale + math.log(-math.expm1(-raw_scale))
    with torch.no_grad():
        model.mano_head.init_cam.copy_(
            torch.tensor([[raw_scale, 0.0, 0.0]], dtype=model.mano_head.init_cam.dtype)
        )
    model.mano_head.register_forward_hook(positive_camera_hook)
    return model, cfg, weight


def masked_mean(values, mask):
    mask = mask.bool()
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(values)
    return torch.where(mask, values, 0.0).sum() / mask.sum().clamp_min(1)


def rotation_angle(predicted, target):
    relative = predicted @ target.transpose(-1, -2)
    cosine = (
        relative.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1) - 1.0
    ) / 2.0
    skew = torch.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        dim=-1,
    )
    sine = 0.5 * torch.linalg.vector_norm(skew, dim=-1)
    return torch.atan2(sine, cosine)


def compute_losses(output, batch, weights):
    predicted_xyz = output["pred_keypoints_3d"]
    predicted_xyz = predicted_xyz - predicted_xyz[:, :1]
    xyz_valid = batch["xyz_valid"]
    uv_valid = batch["uv_valid"]
    xyz_difference = (predicted_xyz - batch["xyz"]).abs()
    uv_difference = (output["pred_keypoints_2d"] - batch["uv"]).abs()
    tip_difference = xyz_difference[:, TIPS]
    pred_bones = torch.stack(
        [predicted_xyz[:, b] - predicted_xyz[:, a] for a, b in EDGES], dim=1
    )
    target_bones = torch.stack(
        [batch["xyz"][:, b] - batch["xyz"][:, a] for a, b in EDGES], dim=1
    )
    cosine = torch.nn.functional.cosine_similarity(pred_bones, target_bones, dim=-1)
    bone_angle = torch.acos(cosine.clamp(-1 + 1e-6, 1 - 1e-6))
    params = output["pred_mano_params"]
    terms = {
        "visible_joints_2d_l1": masked_mean(uv_difference, uv_valid),
        "wrist_relative_joints_3d_l1": masked_mean(xyz_difference, xyz_valid),
        "fingertip_wrist_relative_3d_l1": masked_mean(tip_difference, xyz_valid),
        "bone_direction_angular": masked_mean(bone_angle, xyz_valid),
        "global_orient_geodesic": masked_mean(
            rotation_angle(params["global_orient"][:, 0], batch["global_orient"]),
            batch["global_orient_valid"],
        ),
        "articulated_pose_geodesic": masked_mean(
            rotation_angle(params["hand_pose"], batch["hand_pose"]),
            batch["hand_pose_valid"],
        ),
        "betas_l2": masked_mean(
            (params["betas"] - batch["betas"]).square(), batch["betas_valid"]
        ),
    }
    sample_counts = {
        "visible_joints_2d_l1": int((uv_valid.sum(1) > 0).sum()),
        "wrist_relative_joints_3d_l1": int(xyz_valid.sum()),
        "fingertip_wrist_relative_3d_l1": int(xyz_valid.sum()),
        "bone_direction_angular": int(xyz_valid.sum()),
        "global_orient_geodesic": int(batch["global_orient_valid"].sum()),
        "articulated_pose_geodesic": int(batch["hand_pose_valid"].sum()),
        "betas_l2": int(batch["betas_valid"].sum()),
    }
    scalar_counts = {
        "visible_joints_2d_l1": int(uv_valid.sum()) * 2,
        "wrist_relative_joints_3d_l1": int(xyz_valid.sum()) * 21 * 3,
        "fingertip_wrist_relative_3d_l1": int(xyz_valid.sum()) * len(TIPS) * 3,
        "bone_direction_angular": int(xyz_valid.sum()) * len(EDGES),
        "global_orient_geodesic": sample_counts["global_orient_geodesic"],
        "articulated_pose_geodesic": (
            sample_counts["articulated_pose_geodesic"] * 15
        ),
        "betas_l2": sample_counts["betas_l2"] * 10,
    }
    total = sum(terms[name] * weights[name] for name in terms)
    return total, terms, sample_counts, scalar_counts


def camera_depth(output):
    return output["pred_keypoints_3d"][:, :, 2] + output["pred_cam_t"][:, None, 2]


def make_eval_loader(dataset, count, batch_size, workers):
    indices = np.linspace(0, len(dataset) - 1, min(count, len(dataset))).round().astype(int)
    return DataLoader(
        Subset(dataset, indices.tolist()), batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
        worker_init_fn=worker_init,
    )


def draw_overlay(images, target_uv, predicted_uv, output_path):
    tiles = []
    count = min(8, len(images))
    for index in range(count):
        rgb = images[index].transpose(1, 2, 0)
        rgb = np.ascontiguousarray(
            np.clip((rgb * STD + MEAN) * 255, 0, 255).astype(np.uint8)
        )
        for points, color in (
            (target_uv[index], (0, 255, 0)),
            (predicted_uv[index], (255, 0, 0)),
        ):
            for x, y in (points + 0.5) * rgb.shape[0]:
                cv2.circle(rgb, (round(float(x)), round(float(y))), 2, color, -1)
        tiles.append(rgb)
    contact = np.concatenate(tiles, axis=1)
    cv2.imwrite(str(output_path), cv2.cvtColor(contact, cv2.COLOR_RGB2BGR))


def evaluate(model, loaders, device, output_dir, step):
    model.eval()
    result = {"step": step, "domains": {}}
    with torch.no_grad():
        for domain, loader in loaders.items():
            predicted_xyzs, target_xyzs = [], []
            predicted_uvs, target_uvs, uv_valids = [], [], []
            cameras, depths = [], []
            global_errors, pose_errors, beta_errors = [], [], []
            first_images = None
            for batch in loader:
                images = batch["img"].to(device, non_blocking=True)
                moved = {
                    key: value.to(device, non_blocking=True)
                    if torch.is_tensor(value) else value
                    for key, value in batch.items()
                }
                output = model.forward_step({"img": images})
                predicted_xyz = output["pred_keypoints_3d"]
                predicted_xyz = predicted_xyz - predicted_xyz[:, :1]
                predicted_xyzs.append(predicted_xyz.cpu().numpy())
                target_xyzs.append(batch["xyz"].numpy())
                predicted_uvs.append(output["pred_keypoints_2d"].cpu().numpy())
                target_uvs.append(batch["uv"].numpy())
                uv_valids.append(batch["uv_valid"].numpy())
                cameras.append(output["pred_cam"].cpu().numpy())
                depths.append(camera_depth(output).cpu().numpy())
                params = output["pred_mano_params"]
                if moved["global_orient_valid"].any():
                    values = rotation_angle(
                        params["global_orient"][:, 0], moved["global_orient"]
                    )
                    global_errors.extend(
                        values[moved["global_orient_valid"]].cpu().numpy().tolist()
                    )
                if moved["hand_pose_valid"].any():
                    values = rotation_angle(params["hand_pose"], moved["hand_pose"]).mean(1)
                    pose_errors.extend(
                        values[moved["hand_pose_valid"]].cpu().numpy().tolist()
                    )
                if moved["betas_valid"].any():
                    values = torch.linalg.vector_norm(
                        params["betas"] - moved["betas"], dim=1
                    )
                    beta_errors.extend(
                        values[moved["betas_valid"]].cpu().numpy().tolist()
                    )
                if first_images is None:
                    first_images = batch["img"].numpy()
            predicted_xyz = np.concatenate(predicted_xyzs)
            target_xyz = np.concatenate(target_xyzs)
            predicted_uv = np.concatenate(predicted_uvs)
            target_uv = np.concatenate(target_uvs)
            uv_valid = np.concatenate(uv_valids)
            camera = np.concatenate(cameras)
            depth = np.concatenate(depths)
            error3d = np.linalg.norm(predicted_xyz - target_xyz, axis=-1) * 1000
            error2d = np.linalg.norm(predicted_uv - target_uv, axis=-1) * 256
            span = np.ptp(predicted_uv, axis=1).max(1) * 256
            metrics = {
                "count": len(predicted_xyz),
                "wrist_relative_mpjpe_mm": float(error3d.mean()),
                "fingertip_wrist_relative_error_mm": float(error3d[:, TIPS].mean()),
                "valid_2d_epe_px": float(error2d[uv_valid].mean()),
                "global_orientation_error_deg": (
                    float(np.degrees(global_errors).mean()) if global_errors else None
                ),
                "articulated_pose_error_deg": (
                    float(np.degrees(pose_errors).mean()) if pose_errors else None
                ),
                "betas_l2": float(np.mean(beta_errors)) if beta_errors else None,
                "projection_span_px": {
                    "min": float(span.min()),
                    "median": float(np.median(span)),
                    "max": float(span.max()),
                },
                "weak_camera_scale": {
                    "min": float(camera[:, 0].min()),
                    "median": float(np.median(camera[:, 0])),
                    "max": float(camera[:, 0].max()),
                },
                "nonfinite_rate": float(
                    (~np.isfinite(predicted_xyz).all((1, 2))).mean()
                ),
                "negative_depth_rate": float((~(depth > 0).all(1)).mean()),
                "projection_collapse_rate": float((span < 5).mean()),
            }
            result["domains"][domain] = metrics
            draw_overlay(
                first_images, target_uv[:len(first_images)],
                predicted_uv[:len(first_images)],
                output_dir / f"{domain}_overlay_step_{step:07d}.jpg",
            )
    with (output_dir / "evaluations.jsonl").open("a") as handle:
        handle.write(json.dumps(result, allow_nan=False) + "\n")
    print(json.dumps({"evaluation": result}, allow_nan=False), flush=True)
    return result


def save_checkpoint(path, model, optimizer, step, config):
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": config,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/resnet50_real_mixed_m_s0_subject02.json"
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text())
    if config["stage"] != "M-S0":
        raise ValueError("this entry point currently accepts M-S0 only")
    scope = config["scope"]
    if not scope["final_e4_forbidden"] or not scope["reused_e3_forbidden"]:
        raise ValueError("M-S0 data-exclusion guards are required")
    for path in config["datasets"].values():
        lowered = path.lower()
        if "final_e4" in lowered or "subject-01" in lowered:
            raise ValueError(f"forbidden M-S0 data path: {path}")
    out = ROOT / "artifacts" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, out / config_path.name)
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
    image_size = config["model"]["image_size"]
    augmentation = config["augmentation"]
    paths = {name: ROOT / value for name, value in config["datasets"].items()}
    train_datasets = [
        DexYCBDataset(paths["dexycb"], image_size, augmentation, True),
        FreiHANDDataset(paths["freihand"], image_size, augmentation, True),
        RHDDataset(
            paths["rhd_labels"], paths["rhd_images"], image_size, augmentation, True
        ),
    ]
    eval_datasets = {
        "dexycb": DexYCBDataset(paths["dexycb"], image_size, augmentation, False),
        "freihand": FreiHANDDataset(paths["freihand"], image_size, augmentation, False),
        "rhd": RHDDataset(
            paths["rhd_labels"], paths["rhd_images"], image_size, augmentation, False
        ),
    }
    composition = config["batch_composition"]
    training = config["training"]
    if training["updates"] != 500:
        raise ValueError("M-S0 requires exactly 500 updates")
    expected_composition = {"dexycb": 16, "freihand": 10, "rhd": 4}
    if composition != expected_composition:
        raise ValueError(
            f"M-S0 requires batch composition {expected_composition}, got {composition}"
        )
    counts = [composition[name] for name in ("dexycb", "freihand", "rhd")]
    sampler = FixedMixBatchSampler(
        [len(dataset) for dataset in train_datasets],
        counts, training["updates"], seed,
    )
    train_loader = DataLoader(
        ConcatDataset(train_datasets), batch_sampler=sampler,
        num_workers=training["workers"], pin_memory=True,
        persistent_workers=True, worker_init_fn=worker_init,
    )
    eval_loaders = {
        name: make_eval_loader(
            dataset, training["eval_samples_per_domain"],
            training["eval_batch_size"], training["eval_workers"],
        )
        for name, dataset in eval_datasets.items()
    }
    model, cfg, imagenet_weight = configure_model(config)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.get_parameters(), lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    (out / "effective_config.yaml").write_text(cfg.dump())
    dex_manifest = paths["dexycb"] / "sealed_manifest.json"
    provenance = {
        "run_id": args.run_id,
        "stage": "M-S0",
        "start_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256(config_path),
        "script_sha256": sha256(Path(__file__)),
        "imagenet_weight_sha256": sha256(imagenet_weight),
        "dexycb_manifest_sha256": sha256(dex_manifest),
        "dataset_instances": {
            name: len(dataset)
            for name, dataset in zip(("dexycb", "freihand", "rhd"), train_datasets)
        },
        "batch_composition": composition,
        "model_initialization": (
            "ImageNet1K V2 backbone plus new Fast-HaMeR MANO head; "
            "no RHD checkpoint, teacher, KD or adversarial loss"
        ),
        "label_policy": config["label_policy"],
        "not_for_accuracy_conclusions": True,
    }
    write_json(out / "provenance.json", provenance)
    snapshots = set(training["snapshots"])
    step = 0
    initial_backbone = next(model.backbone.parameters()).detach().cpu().clone()
    initial_head = model.mano_head.decpose.weight.detach().cpu().clone()
    evaluate(model, eval_loaders, device, out, step)
    save_checkpoint(out / "step_0000000.pt", model, optimizer, step, config)
    cumulative_sample_counts = {name: 0 for name in config["losses"]}
    cumulative_scalar_counts = {name: 0 for name in config["losses"]}
    gradient_maxima = {
        name: {
            "head_decpose": 0.0, "head_decshape": 0.0,
            "head_deccam": 0.0, "backbone_first": 0.0,
        }
        for name in config["losses"]
    }
    start = time.perf_counter()
    for batch in train_loader:
        model.train()
        moved = {
            key: value.to(device, non_blocking=True)
            if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        optimizer.zero_grad(set_to_none=True)
        output = model.forward_step({"img": moved["img"]}, train=True)
        total, terms, sample_counts_now, scalar_counts_now = compute_losses(
            output, moved, config["losses"]
        )
        for name in config["losses"]:
            cumulative_sample_counts[name] += sample_counts_now[name]
            cumulative_scalar_counts[name] += scalar_counts_now[name]
        depth = camera_depth(output)
        span = torch.max(
            torch.max(output["pred_keypoints_2d"], dim=1).values
            - torch.min(output["pred_keypoints_2d"], dim=1).values,
            dim=1,
        ).values * image_size
        if (
            not torch.isfinite(total)
            or not torch.isfinite(depth).all()
            or not torch.isfinite(output["pred_keypoints_2d"]).all()
        ):
            raise FloatingPointError(f"nonfinite at step {step + 1}")
        gradients = None
        if step == 0 or (step + 1) % training["gradient_log_interval"] == 0:
            targets = (
                model.mano_head.decpose.weight,
                model.mano_head.decshape.weight,
                model.mano_head.deccam.weight,
                next(model.backbone.parameters()),
            )
            gradients = {}
            for name, term in terms.items():
                grads = torch.autograd.grad(
                    term * config["losses"][name], targets,
                    retain_graph=True, allow_unused=True,
                )
                gradients[name] = {
                    "head_decpose": (
                        None if grads[0] is None else float(grads[0].norm())
                    ),
                    "head_decshape": (
                        None if grads[1] is None else float(grads[1].norm())
                    ),
                    "head_deccam": (
                        None if grads[2] is None else float(grads[2].norm())
                    ),
                    "backbone_first": (
                        None if grads[3] is None else float(grads[3].norm())
                    ),
                }
                for target_name, value in gradients[name].items():
                    if value is not None:
                        gradient_maxima[name][target_name] = max(
                            gradient_maxima[name][target_name], value
                        )
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.get_parameters(), training["clip_grad_norm"],
            error_if_nonfinite=True,
        )
        optimizer.step()
        step += 1
        domain_counts = torch.bincount(
            moved["dataset_id"], minlength=len(DATASET_IDS)
        ).cpu().tolist()
        if domain_counts != counts:
            raise AssertionError(
                f"domain batch composition changed: {domain_counts} != {counts}"
            )
        row = {
            "step": step,
            "loss": float(total.detach()),
            "loss_components": {
                name: float(value.detach()) for name, value in terms.items()
            },
            "effective_sample_counts": sample_counts_now,
            "effective_scalar_counts": scalar_counts_now,
            "cumulative_effective_sample_counts": dict(cumulative_sample_counts),
            "cumulative_effective_scalar_counts": dict(cumulative_scalar_counts),
            "domain_batch_counts": {
                name: domain_counts[index]
                for name, index in DATASET_IDS.items()
            },
            "component_weighted_gradient_norms": gradients,
            "total_gradient_norm_before_clip": float(grad_norm),
            "weak_camera_scale_min_max": [
                float(output["pred_cam"][:, 0].min()),
                float(output["pred_cam"][:, 0].max()),
            ],
            "camera_depth_min_max_m": [
                float(depth.min()), float(depth.max())
            ],
            "projection_span_px_min_median_max": [
                float(span.min()), float(span.median()), float(span.max())
            ],
            "elapsed_s": time.perf_counter() - start,
        }
        with (out / "steps.jsonl").open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        if step <= 5 or step % 25 == 0:
            print(json.dumps(row, allow_nan=False), flush=True)
        if step in snapshots:
            evaluate(model, eval_loaders, device, out, step)
            save_checkpoint(
                out / f"step_{step:07d}.pt", model, optimizer, step, config
            )
    save_checkpoint(out / "last.pt", model, optimizer, step, config)
    model.eval()
    reload_sample = eval_datasets["freihand"][0]["img"][None].to(device)
    with torch.no_grad():
        before = model.forward_step({"img": reload_sample})
    restored = HAMER(cfg, init_renderer=False)
    restored.mano_head.register_forward_hook(positive_camera_hook)
    checkpoint = torch.load(out / "last.pt", map_location="cpu")
    restored.load_state_dict(checkpoint["state_dict"], strict=True)
    restored.to(device).eval()
    with torch.no_grad():
        after = restored.forward_step({"img": reload_sample})
    reload_xyz = float(
        (before["pred_keypoints_3d"] - after["pred_keypoints_3d"]).abs().max()
    )
    reload_uv = float(
        (before["pred_keypoints_2d"] - after["pred_keypoints_2d"]).abs().max()
    )
    if reload_xyz > 1e-7 or reload_uv > 1e-7:
        raise AssertionError(
            f"strict reload mismatch: xyz={reload_xyz}, uv={reload_uv}"
        )
    backbone_update = float(
        (next(restored.backbone.parameters()).detach().cpu() - initial_backbone)
        .abs().max()
    )
    head_update = float(
        (restored.mano_head.decpose.weight.detach().cpu() - initial_head).abs().max()
    )
    final = evaluate(restored, eval_loaders, device, out, step)
    passed = (
        step == training["updates"]
        and reload_xyz <= 1e-7
        and reload_uv <= 1e-7
        and backbone_update > 0
        and head_update > 0
        and all(
            metrics["nonfinite_rate"] == 0
            and metrics["negative_depth_rate"] == 0
            and metrics["projection_span_px"]["median"] > 20
            and metrics["projection_collapse_rate"] == 0
            for metrics in final["domains"].values()
        )
        and all(value > 0 for value in cumulative_sample_counts.values())
        and all(value > 0 for value in cumulative_scalar_counts.values())
        and all(
            values["backbone_first"] > 0
            and max(
                values["head_decpose"],
                values["head_decshape"],
                values["head_deccam"],
            ) > 0
            for values in gradient_maxima.values()
        )
    )
    results = {
        "stage": "M-S0",
        "passed": bool(passed),
        "actual_updates": step,
        "final": final,
        "cumulative_effective_sample_counts": cumulative_sample_counts,
        "cumulative_effective_scalar_counts": cumulative_scalar_counts,
        "maximum_component_weighted_gradient_norms": gradient_maxima,
        "strict_reload_xyz_max_m": reload_xyz,
        "strict_reload_uv_max_normalized": reload_uv,
        "backbone_first_parameter_max_update": backbone_update,
        "head_decpose_max_update": head_update,
        "last_checkpoint_sha256": sha256(out / "last.pt"),
        "end_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(out / "results.json", results)
    if not passed:
        raise AssertionError("M-S0 acceptance criteria failed; inspect results.json")
    print(json.dumps({"completed": results}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
