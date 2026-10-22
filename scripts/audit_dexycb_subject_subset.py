"""Independently audit 64 views from an extracted DexYCB subject subset."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml


EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
COLORS = [
    (70, 70, 240), (50, 210, 240), (70, 210, 90),
    (230, 150, 40), (220, 70, 210),
]


def stats(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def camera_matrix(root, serial):
    path = root / "calibration" / "intrinsics" / f"{serial}_640x480.yml"
    raw = yaml.load(path.read_text(), Loader=yaml.FullLoader)["color"]
    return np.asarray([
        [raw["fx"], 0.0, raw["ppx"]],
        [0.0, raw["fy"], raw["ppy"]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def draw(image, target, projected, valid):
    result = image.copy()
    for index, (a, b) in enumerate(EDGES):
        if not (valid[a] and valid[b]):
            continue
        color = COLORS[min(index // 4, 4)]
        cv2.line(
            result, tuple(np.rint(target[a]).astype(int)),
            tuple(np.rint(target[b]).astype(int)), color, 2, cv2.LINE_AA,
        )
    for point in target[valid]:
        cv2.circle(
            result, tuple(np.rint(point).astype(int)), 3,
            (40, 240, 40), -1, cv2.LINE_AA,
        )
    for point in projected[valid]:
        cv2.circle(
            result, tuple(np.rint(point).astype(int)), 4,
            (240, 40, 240), 1, cv2.LINE_AA,
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    manifest = json.loads((args.root / "sealed_manifest.json").read_text())
    views = []
    for sample in manifest["samples"]:
        for view in sample["views"]:
            views.append({
                "sample_id": sample["sample_id"],
                "subject": sample["subject"],
                "sequence": sample["sequence"],
                "frame_index": sample["frame_index"],
                "mano_side": sample["mano_side"],
                **view,
            })
    if args.count > len(views):
        raise ValueError(f"requested {args.count} unique views from only {len(views)}")
    indices = np.linspace(0, len(views) - 1, args.count).round().astype(int)
    if len(np.unique(indices)) != args.count:
        raise ValueError("audit view selection produced duplicate indices")
    rows = []
    overlays = []
    projection_errors = []
    roundtrip_errors = []
    spans = []
    sides = {}
    matrices = {}
    bad_dir = args.output / "bad_cases"
    for index in indices:
        row = views[index]
        serial = row["camera_serial"]
        if serial not in matrices:
            matrices[serial] = camera_matrix(args.root, serial)
        labels = np.load(args.root / row["label"])
        image = cv2.imread(str(args.root / row["image"]))
        if image is None:
            raise FileNotFoundError(args.root / row["image"])
        xyz = labels["joint_3d"][0].astype(np.float64)
        target = labels["joint_2d"][0].astype(np.float64)
        homogeneous = xyz @ matrices[serial].T
        projected = homogeneous[:, :2] / homogeneous[:, 2:]
        valid = (
            np.isfinite(xyz).all(1)
            & np.isfinite(target).all(1)
            & (target >= 0).all(1)
        )
        if valid.sum() < 4:
            raise ValueError(f"fewer than 4 valid joints in {row['label']}")
        error = np.linalg.norm(projected[valid] - target[valid], axis=1)
        lo, hi = target[valid].min(0), target[valid].max(0)
        center = (lo + hi) / 2
        side = max(float((hi - lo).max()) * 2.0, 64.0)
        scale = 256.0 / side
        affine = np.asarray([
            [scale, 0.0, 128.0 - scale * center[0]],
            [0.0, scale, 128.0 - scale * center[1]],
        ], np.float64)
        crop_target = np.c_[target, np.ones(21)] @ affine.T
        crop_projected = np.c_[projected, np.ones(21)] @ affine.T
        recovered = (
            np.c_[crop_target, np.ones(21)]
            @ cv2.invertAffineTransform(affine).T
        )
        roundtrip = np.linalg.norm(recovered[valid] - target[valid], axis=1)
        crop = cv2.warpAffine(
            image, affine.astype(np.float32), (256, 256),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        )
        rendered = draw(crop, crop_target, crop_projected, valid)
        cv2.putText(
            rendered, f"{index:05d} {serial[-4:]}", (4, 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA,
        )
        overlays.append(rendered)
        positive_depth = bool((xyz[:, 2] > 0).all())
        finite = bool(
            np.isfinite(xyz).all()
            and np.isfinite(target[valid]).all()
            and np.isfinite(labels["pose_m"]).all()
        )
        pose_m_valid = bool(np.any(np.abs(labels["pose_m"]) > 1e-8))
        passed = bool(
            finite and positive_depth and error.max() < 2.0
            and roundtrip.max() < 1e-3
        )
        if not passed:
            bad_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(bad_dir / f"{index:05d}_{serial}.jpg"), rendered)
        projection_errors.extend(error.tolist())
        roundtrip_errors.extend(roundtrip.tolist())
        spans.append(float(np.ptp(crop_target, axis=0).max()))
        sides[row["mano_side"]] = sides.get(row["mano_side"], 0) + 1
        rows.append({
            "view_index": int(index),
            "sample_id": row["sample_id"],
            "subject": row["subject"],
            "sequence": row["sequence"],
            "frame_index": row["frame_index"],
            "camera_serial": serial,
            "mano_side": row["mano_side"],
            "valid_joint_count": int(valid.sum()),
            "finite": finite,
            "all_positive_depth": positive_depth,
            "pose_m_valid": pose_m_valid,
            "projection_error_px": stats(error),
            "crop_full_roundtrip_error_px": stats(roundtrip),
            "crop_projection_span_px": spans[-1],
            "global_orient_axis_angle": labels["pose_m"][0, :3].tolist(),
            "passed": passed,
        })
    grid_side = int(np.ceil(np.sqrt(len(overlays))))
    contact = np.full((grid_side * 256, grid_side * 256, 3), 25, np.uint8)
    for position, image in enumerate(overlays):
        y, x = divmod(position, grid_side)
        contact[y * 256:(y + 1) * 256, x * 256:(x + 1) * 256] = image
    cv2.imwrite(str(args.output / "contact_64.jpg"), contact)
    report = {
        "dataset": "DexYCB",
        "subject": manifest["subject"],
        "split": manifest["split"],
        "available_views": len(views),
        "audited_views": len(rows),
        "unique_audited_views": len(set(row["view_index"] for row in rows)),
        "camera_serials": manifest["camera_serials"],
        "mano_side_counts": sides,
        "projection_error_px": stats(projection_errors),
        "crop_full_roundtrip_error_px": stats(roundtrip_errors),
        "crop_projection_span_px": stats(spans),
        "nonfinite_views": sum(not row["finite"] for row in rows),
        "nonpositive_depth_views": sum(
            not row["all_positive_depth"] for row in rows
        ),
        "missing_pose_m_views": sum(not row["pose_m_valid"] for row in rows),
        "bad_case_count": sum(not row["passed"] for row in rows),
        "joint_convention": (
            "wrist; thumb 1-4; index 5-8; middle 9-12; "
            "ring 13-16; little 17-20"
        ),
        "left_hand_policy": (
            "image and x coordinates are mirrored to the right-hand canonical "
            "frame; rotation matrices use F R F. This subject audit contains "
            f"the side counts recorded above: {sides}."
        ),
        "m_s0_supervision_scope": {
            "metric_3d_joints": True,
            "camera_projected_2d_joints": True,
            "global_orientation": True,
            "articulated_pose": (
                "masked pending independent DexYCB PCA-to-axis-angle audit"
            ),
            "betas": "masked pending calibration convention audit",
        },
        "rows": rows,
        "passed": all(row["passed"] for row in rows),
    }
    (args.output / "audit.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    if not report["passed"]:
        raise AssertionError("DexYCB subject subset audit failed")
    print(json.dumps({
        "passed": True,
        "audited_views": len(rows),
        "projection_error_px": report["projection_error_px"],
        "crop_full_roundtrip_error_px": report[
            "crop_full_roundtrip_error_px"
        ],
    }, allow_nan=False))


if __name__ == "__main__":
    main()
