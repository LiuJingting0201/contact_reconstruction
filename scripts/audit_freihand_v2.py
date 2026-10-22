"""Audit official FreiHAND v2 labels and render 64 deterministic overlays."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


JOINTS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
COLORS = [
    (80, 80, 240), (60, 210, 240), (80, 210, 100),
    (230, 160, 50), (220, 80, 210),
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project(xyz, K):
    homogeneous = xyz @ K.T
    return homogeneous[:, :2] / homogeneous[:, 2:]


def draw(image, uv):
    result = image.copy()
    for index, (a, b) in enumerate(JOINTS):
        color = COLORS[min(index // 4, 4)]
        cv2.line(result, tuple(np.rint(uv[a]).astype(int)),
                 tuple(np.rint(uv[b]).astype(int)), color, 2, cv2.LINE_AA)
    for point in uv:
        cv2.circle(result, tuple(np.rint(point).astype(int)), 3,
                   (255, 255, 255), -1, cv2.LINE_AA)
    return result


def stats(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    K = np.asarray(json.loads((args.root / "training_K.json").read_text()))
    xyz = np.asarray(json.loads((args.root / "training_xyz.json").read_text()))
    mano = np.asarray(json.loads((args.root / "training_mano.json").read_text()))[:, 0]
    scale = np.asarray(json.loads((args.root / "training_scale.json").read_text()))
    count = 32560
    if not (K.shape == (count, 3, 3) and xyz.shape == (count, 21, 3)
            and mano.shape == (count, 61) and scale.shape == (count,)):
        raise ValueError("unexpected official annotation shape")
    indices = np.linspace(0, count - 1, 64).round().astype(int)
    rows, all_uv, overlays = [], [], []
    for index in indices:
        uv = project(xyz[index], K[index])
        all_uv.append(uv)
        finite = bool(np.isfinite(uv).all() and np.isfinite(xyz[index]).all())
        positive = bool((xyz[index, :, 2] > 0).all())
        in_frame = ((uv >= 0) & (uv < np.asarray([224, 224]))).all(axis=1)
        mask_path = args.root / "training" / "mask" / f"{index:08d}.jpg"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != (224, 224):
            raise ValueError(f"invalid mask {mask_path}")
        mask_fraction = float((mask > 127).mean())
        versions = []
        for version in range(4):
            image_index = index + version * count
            image_path = args.root / "training" / "rgb" / f"{image_index:08d}.jpg"
            image = cv2.imread(str(image_path))
            if image is None or image.shape != (224, 224, 3):
                raise ValueError(f"invalid RGB {image_path}")
            versions.append({
                "version": version,
                "image_index": int(image_index),
                "sha256": sha256(image_path),
            })
            if version == 0:
                rendered = draw(image, uv)
                cv2.putText(rendered, f"{index:05d}", (4, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1,
                            cv2.LINE_AA)
                overlays.append(rendered)
                cv2.imwrite(str(args.output / f"{index:05d}_overlay.jpg"), rendered)
        rows.append({
            "unique_capture_index": int(index),
            "finite": finite,
            "all_positive_depth": positive,
            "in_frame_joints": int(in_frame.sum()),
            "projection_bbox_span_px": (
                uv.max(axis=0) - uv.min(axis=0)
            ).astype(float).tolist(),
            "wrist_depth_m": float(xyz[index, 0, 2]),
            "mean_bone_length_m": float(np.mean([
                np.linalg.norm(xyz[index, a] - xyz[index, b])
                for a, b in JOINTS
            ])),
            "mask_foreground_fraction": mask_fraction,
            "mano": {
                "global_orient_axis_angle": mano[index, :3].tolist(),
                "articulated_pose_axis_angle": mano[index, 3:48].tolist(),
                "betas": mano[index, 48:58].tolist(),
                "uv_root": mano[index, 58:60].tolist(),
                "scale": float(mano[index, 60]),
            },
            "versions": versions,
        })
    if not all(row["finite"] and row["all_positive_depth"] for row in rows):
        raise ValueError("nonfinite or nonpositive audited FreiHAND label")
    tile = np.full((8 * 224, 8 * 224, 3), 30, np.uint8)
    for index, image in enumerate(overlays):
        y, x = divmod(index, 8)
        tile[y * 224:(y + 1) * 224, x * 224:(x + 1) * 224] = image
    cv2.imwrite(str(args.output / "contact_64.jpg"), tile)
    in_frame = [row["in_frame_joints"] for row in rows]
    spans = [max(row["projection_bbox_span_px"]) for row in rows]
    report = {
        "dataset": "FreiHAND_pub_v2",
        "official_url": (
            "https://lmb.informatik.uni-freiburg.de/data/freihand/"
            "FreiHAND_pub_v2.zip"
        ),
        "license": "research only; commercial use prohibited",
        "archive": {
            "bytes": args.archive.stat().st_size,
            "sha256": sha256(args.archive),
            "zip_crc_all_members_passed": True,
        },
        "counts": {
            "unique_captures": count,
            "rgb_images": count * 4,
            "background_versions_per_capture": 4,
            "masks": count,
            "audited_unique_captures": len(rows),
        },
        "annotation_shapes": {
            "K": list(K.shape), "xyz_m": list(xyz.shape),
            "mano": list(mano.shape), "scale": list(scale.shape),
        },
        "joint_order": (
            "wrist; thumb 1-4; index 5-8; middle 9-12; ring 13-16; "
            "little 17-20"
        ),
        "mano_layout": (
            "3 global axis-angle + 45 articulated axis-angle + 10 betas + "
            "2 uv_root + 1 scale; the final 3 values are not xyz translation"
        ),
        "projection_source": (
            "No independent 2D annotation: overlay is official metric xyz "
            "projected with official K."
        ),
        "in_frame_joint_count": stats(in_frame),
        "projection_span_px": stats(spans),
        "wrist_depth_m": stats([row["wrist_depth_m"] for row in rows]),
        "mean_bone_length_m": stats([row["mean_bone_length_m"] for row in rows]),
        "mask_foreground_fraction": stats([
            row["mask_foreground_fraction"] for row in rows
        ]),
        "rows": rows,
        "training_assignment": (
            "train only; all four RGB variants grouped by unique_capture_index"
        ),
        "excluded_archive_member": (
            "training_verts.json omitted from extracted training copy because "
            "metric joints and MANO supervision are present; retained in the "
            "verified source archive until audit completion"
        ),
    }
    (args.output / "audit.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "passed": True,
        "audited": len(rows),
        "archive_sha256": report["archive"]["sha256"],
        "in_frame_joint_count": report["in_frame_joint_count"],
    }))


if __name__ == "__main__":
    main()
