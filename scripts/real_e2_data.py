"""Strict, dataset-neutral real stereo input adapter. No guessed calibration."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

JOINT_ORDER = ["wrist"] + [
    f"{finger}_{joint}" for finger in ("thumb", "index", "middle", "ring", "little")
    for joint in ("mcp", "pip", "dip", "tip")
]


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def numeric(value, shape, name):
    x = np.asarray(value, dtype=np.float64)
    require(x.shape == shape and np.isfinite(x).all(), f"{name}: expected finite {shape}")
    return x


def transform(xyz, T):
    return xyz @ T[:3, :3].T + T[:3, 3]


def project(xyz, K, T):
    camera = transform(xyz, T)
    homogeneous = camera @ K.T
    return homogeneous[:, :2] / homogeneous[:, 2:], camera


def validate_geometry(K, T, xyz, uv, valid, baseline_range=(.005, 3.), tolerance=2.):
    """Canonical undistorted pixels and world-to-camera transforms; arbitrary rig."""
    K = numeric(K, (2, 3, 3), "K")
    T = numeric(T, (2, 4, 4), "T_world_to_camera")
    xyz = numeric(xyz, (21, 3), "GT world XYZ meters")
    uv = numeric(uv, (2, 21, 2), "GT2D")
    valid = np.asarray(valid, bool)
    require(valid.shape == (2, 21) and (valid.sum(-1) >= 4).all(), "insufficient GT2D")
    for k, t in zip(K, T):
        require(np.allclose(k[2], [0, 0, 1]) and k[0, 0] > 0 and k[1, 1] > 0, "invalid K")
        require(np.allclose(t[3], [0, 0, 0, 1]), "invalid homogeneous transform")
        require(np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-5)
                and np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-5), "improper rotation")
    centers = np.array([np.linalg.inv(t)[:3, 3] for t in T])
    baseline = np.linalg.norm(centers[1] - centers[0])
    require(baseline_range[0] <= baseline <= baseline_range[1], "zero/implausible baseline")
    errors, depths = [], []
    for v in range(2):
        projected, camera = project(xyz, K[v], T[v])
        errors.append(np.linalg.norm(projected - uv[v], axis=-1)[valid[v]])
        depths.append(camera[:, 2].min())
        require(camera[:, 2].min() > 0, "GT behind camera: check units/extrinsic direction")
        require(errors[-1].max() <= tolerance, "GT3D/GT2D mismatch: check calibration, units, synchronization")
    common = valid.all(0)
    require(common.sum() >= 4, "need four common GT points for stereo cheirality")
    # Raw uL-uR has no universal sign for nonrectified/convergent cameras.
    # Triangulation + positive depth checks disparity in the actual calibrated rig.
    p0, p1 = K[0] @ T[0, :3], K[1] @ T[1, :3]
    homogeneous = cv2.triangulatePoints(p0, p1, uv[0, common].T, uv[1, common].T)
    require(np.all(abs(homogeneous[3]) > 1e-10), "degenerate stereo rays")
    tri = (homogeneous[:3] / homogeneous[3]).T
    require(np.isfinite(tri).all(), "nonfinite triangulation")
    require(all((transform(tri, t)[:, 2] > 0).all() for t in T), "wrong disparity/cheirality")
    return {"baseline_m": float(baseline), "gt_projection_max_px": float(max(x.max() for x in errors)),
            "gt_minimum_depth_m": float(min(depths)), "triangulated_gt_mean_error_m":
            float(np.linalg.norm(tri - xyz[common], axis=-1).mean()),
            "disparity_validation": "positive triangulated depth in both arbitrary calibrated cameras"}


def crop_affine(box, handedness, resolution=256):
    x1, y1, x2, y2 = numeric(box, (4,), "crop_box_xyxy")
    require(x2 > x1 and y2 > y1, "empty crop")
    side = max(x2 - x1, y2 - y1)
    scale = resolution / side
    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    affine = np.array([[scale, 0, resolution / 2 - scale * center[0]],
                       [0, scale, resolution / 2 - scale * center[1]], [0, 0, 1.]])
    require(handedness in ("left", "right"), "handedness must be explicit")
    if handedness == "left":
        # One full-image-to-canonical-crop reflection, exactly centered on 128.
        affine = np.array([[-1, 0, resolution], [0, 1, 0], [0, 0, 1.]]) @ affine
    require(np.allclose(np.linalg.inv(affine) @ affine, np.eye(3), atol=1e-10), "crop roundtrip")
    return affine


def read_view(view, base, handedness):
    path = (base / view["image"]).resolve()
    require(path.is_file(), f"missing RGB: {path}")
    require(sha256(path) == view["image_sha256"], f"RGB hash mismatch: {path}")
    image = cv2.imread(str(path))
    require(image is not None, "RGB decode failed")
    resolution = [image.shape[1], image.shape[0]]
    require(view["resolution_wh"] == resolution == view["K_resolution_wh"], "K/image resolution mismatch")
    K = numeric(view["K"], (3, 3), "K")
    require(abs(K[0, 1]) < 1e-8 and abs(K[1, 0]) < 1e-8, "skewed K not supported")
    require(0 <= K[0, 2] < resolution[0] and 0 <= K[1, 2] < resolution[1], "principal point outside image")
    distortion = view["distortion"]
    require(bool(distortion["evidence"]), "missing distortion/undistortion evidence")
    uv = numeric(view["gt2d"], (21, 2), "GT2D")
    mode = distortion["state"]
    if mode == "dataset_delivered_pinhole":
        require(view["undistort_count"] == 0, "dataset pinhole pixels must not be corrected again")
        require(distortion["model"] == "dataset_pinhole" and distortion["coefficients"] is None,
                "do not invent raw lens coefficients for delivered pinhole pixels")
        count = 0
    elif mode == "undistorted":
        require(view["undistort_count"] in (0, 1), "duplicate undistortion")
        require(distortion["model"] == "none" and not any(distortion["coefficients"]),
                "undistorted coordinates must have no residual distortion")
        count = view["undistort_count"]
    elif mode == "raw":
        require(view["undistort_count"] == 0, "raw image already undistorted")
        require(distortion["model"] == "opencv_brown5", "unsupported/unspecified distortion model")
        coeff = numeric(distortion["coefficients"], (5,), "k1,k2,p1,p2,k3")
        image = cv2.undistort(image, K, coeff, None, K)
        uv = cv2.undistortPoints(uv[:, None], K, coeff, P=K)[:, 0]
        count = 1
    else:
        raise ValueError("unknown distortion state; do not assume zero")
    require(view["crop_coordinate_system"] == "undistorted_full_image", "crop must match processed image")
    require(view["crop_source"] in ("manual_rgb", "predicted_detector"), "GT crop not deployable")
    require(view["mirror_count"] == 0, "input must not be pre-mirrored")
    A = crop_affine(view["crop_box_xyxy"], handedness)
    corners = np.array([[0, 0, 1], [resolution[0] - 1, resolution[1] - 1, 1]], float)
    require(np.allclose((corners @ A.T) @ np.linalg.inv(A).T, corners), "full/crop roundtrip")
    return {"image": image, "K": K, "uv": uv, "A": A, "undistort_count": count,
            "mirror_count": int(handedness == "left"), "path": str(path)}


def load_manifest(path, config, overlays=None):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    require(manifest["schema"] == "real_e2_v1" and manifest["kind"] == "real", "real input required")
    require(manifest["units"] == "m" and manifest["gt_coordinate_system"] in ("world", "per_camera"),
            "GT must be meters in a declared frame")
    require(manifest["joint_order"] == JOINT_ORDER, "joint order mismatch (wrist, thumb to little)")
    require(manifest["source"]["license_allows_experiment"] is True
            and bool(manifest["source"]["license_url"]), "unverified data license")
    require(manifest["source"]["gt_independent_of_teacher"] is True
            and bool(manifest["source"]["gt_method"]), "missing independent GT provenance")
    rows = manifest["samples"]
    require(20 <= len(rows) <= 50, "first method validation requires 20–50 synchronized times")
    sample_ids, sync_ids, sequence_splits, loaded, reports = set(), set(), {}, [], []
    for row in rows:
        require(row["sample_id"] not in sample_ids, "duplicate sample_id")
        sample_ids.add(row["sample_id"])
        key = (row["sequence_id"], row["sync_id"])
        require(key not in sync_ids, "duplicate synchronized time")
        sync_ids.add(key)
        require(sequence_splits.setdefault(row["sequence_id"], row["split"]) == row["split"],
                "same sequence appears in multiple splits")
        require(bool(row["source_original_id"]) and bool(row["sync_evidence"]), "missing original/sync provenance")
        timestamp = float(row["timestamp_s"])
        require(np.isfinite(timestamp), "nonfinite timestamp")
        views = row["views"]
        require(len(views) == 2 and [v["role"] for v in views] == ["left", "right"], "need ordered paired views")
        require(views[0]["image"] != views[1]["image"] and
                views[0]["image_sha256"] != views[1]["image_sha256"], "same image used twice")
        require(all(v["sync_id"] == row["sync_id"] for v in views), "synchronization ID mismatch")
        if manifest.get("timestamp_kind") == "dataset_frame_index_over_fps":
            require(bool(manifest.get("synchronization_code_evidence")), "missing official sync evidence")
            require(all(v["frame_index"] == row["frame_index"] for v in views), "different frame indices")
            require(views[0]["camera_serial"] != views[1]["camera_serial"], "same camera")
            require(all(float(v["timestamp_s"]) == timestamp for v in views), "different nominal frame times")
        else:
            tolerance = float(manifest["sync_tolerance_s"])
            require(0 <= tolerance <= .005, "sync tolerance must be explicit and <=5ms")
            require(all(abs(float(v["timestamp_s"]) - timestamp) <= tolerance for v in views), "wrong time")
            require(abs(float(views[0]["timestamp_s"]) - float(views[1]["timestamp_s"])) <= tolerance,
                    "left/right timestamps exceed tolerance")
        loaded_views = [read_view(v, path.parent, row["handedness"]) for v in views]
        K = np.stack([v["K"] for v in loaded_views])
        T = numeric([v["T_world_to_camera"] for v in views], (2, 4, 4), "extrinsics")
        per_camera = manifest["gt_coordinate_system"] == "per_camera"
        gt_camera = (numeric([v["gt3d"] for v in views], (2, 21, 3), "per-camera GT3D")
                     if per_camera else None)
        # Internal shared reference may be left camera. No extra dataset world GT
        # is requested or fabricated; keep both delivered camera GT arrays.
        xyz = (transform(gt_camera[0], np.linalg.inv(T[0])) if per_camera
               else numeric(row["gt3d"], (21, 3), "GT3D"))
        uv = np.stack([v["uv"] for v in loaded_views])
        valid = np.array([v["gt2d_valid"] for v in views], bool)
        report = validate_geometry(K, T, xyz, uv, valid, config["baseline_range_m"],
                                   config["gt_projection_tolerance_px"])
        if per_camera:
            for v in range(2):
                direct, _ = project(gt_camera[v], K[v], np.eye(4))
                error = np.linalg.norm(direct-uv[v], axis=-1)[valid[v]]
                require(error.max() <= config["gt_projection_tolerance_px"], "camera GT projection mismatch")
            right_in_shared = transform(gt_camera[1], np.linalg.inv(T[1]))
            cross_error = np.linalg.norm(right_in_shared-xyz, axis=-1)
            require(cross_error.max() < .005, "GT cross-camera extrinsics/unit mismatch >5mm")
            report["right_gt_to_left_mean_error_m"] = float(cross_error.mean())
            report["right_gt_to_left_max_error_m"] = float(cross_error.max())
        else:
            gt_camera = np.stack([transform(xyz, t) for t in T])
        require(row["gt3d_valid"] == [True] * 21, "all 21 independent GT3D joints required for this matrix")
        report.update(sample_id=row["sample_id"], sync_evidence=row["sync_evidence"],
                      timestamp_delta_s=abs(float(views[0]["timestamp_s"]) - float(views[1]["timestamp_s"])),
                      gt2d_origin=[v["gt2d_origin"] for v in views],
                      undistort_counts=[v["undistort_count"] for v in loaded_views],
                      mirror_counts=[v["mirror_count"] for v in loaded_views])
        report["timestamp_kind"] = manifest.get("timestamp_kind", "measured")
        if manifest.get("timestamp_kind") == "dataset_frame_index_over_fps":
            report["timestamp_delta_s"] = None
            report["sync_limit"] = "same official synchronized frame; physical time difference not supplied"
        if overlays is not None:
            tiles = []
            for v in range(2):
                tile = loaded_views[v]["image"].copy()
                projected, _ = project(xyz, K[v], T[v])
                for j in np.flatnonzero(valid[v]):
                    cv2.circle(tile, tuple(np.rint(uv[v, j]).astype(int)), 3, (0, 255, 0), -1)
                    cv2.circle(tile, tuple(np.rint(projected[j]).astype(int)), 5, (255, 0, 255), 1)
                tiles.append(tile)
            h = max(t.shape[0] for t in tiles)
            tiles = [cv2.copyMakeBorder(t, 0, h-t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
            output = Path(overlays) / f"{len(loaded):03d}_gt.jpg"
            output.parent.mkdir(parents=True, exist_ok=True)
            require(cv2.imwrite(str(output), np.concatenate(tiles, axis=1)), "overlay write failed")
        loaded.append({"row": row, "views": loaded_views, "K": K, "T": T,
                       "gt": xyz, "gt_camera": gt_camera, "gt_uv": uv, "valid": valid})
        reports.append(report)
    return manifest, loaded, reports


def loss_numpy(uv, target, confidence, view_weights, robust=False, scale=3.):
    """Reference loss for interface regression, independent of torch backend."""
    confidence = np.asarray(confidence, float)
    require(np.isfinite(confidence).all() and (confidence >= 0).all(), "invalid confidence")
    active = confidence.sum(-1) > 0
    weights = np.asarray(view_weights) * active
    require(weights.sum() > 0, "no active view")
    residual = np.where(confidence[..., None] > 0, uv - target, 0)
    d2 = np.square(residual / scale).sum(-1)
    cost = np.where(d2 <= 1, d2, 2 * np.sqrt(d2) - 1) if robust else d2
    per_view = (cost * confidence).sum(-1) / np.maximum(confidence.sum(-1), 1e-12)
    return float((per_view * weights).sum() / weights.sum())
