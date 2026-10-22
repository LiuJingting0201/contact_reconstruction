"""Strict calibrated-camera input for stereo experiments; NumPy only.

Does not silently alter scale, intrinsics, handedness, or world-to-camera pose.
This adapter is not yet integrated with the complete Dyn-HaMR pipeline.
"""
from pathlib import Path
import numpy as np


def load_calibrated_camera(path, image_size, expected_frames=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        width, height = int(data["width"]), int(data["height"])
        if (width, height) != tuple(image_size):
            raise ValueError("calibration_resolution_mismatch")
        if "intrins" not in data:
            raise ValueError("explicit_intrinsics_required")
        intr = np.array(data["intrins"], dtype=np.float64)
        w2c = np.array(data["w2c"], dtype=np.float64)
    if w2c.ndim != 3 or w2c.shape[1:] != (4, 4):
        raise ValueError("w2c_shape")
    n = len(w2c)
    if n == 0 or intr.shape != (n, 4) or (expected_frames is not None and n != expected_frames):
        raise ValueError("frame_count_or_intrinsics_shape")
    if not np.isfinite(w2c).all() or not np.isfinite(intr).all():
        raise ValueError("nonfinite_calibration")
    if (intr[:, :2] <= 0).any():
        raise ValueError("nonpositive_focal")
    if not np.allclose(w2c[:, 3, :], [0, 0, 0, 1], atol=1e-7, rtol=0):
        raise ValueError("invalid_homogeneous_row")
    rotation = w2c[:, :3, :3]
    if not np.allclose(rotation @ rotation.transpose(0, 2, 1), np.eye(3), atol=1e-5, rtol=0):
        raise ValueError("nonorthogonal_rotation")
    if not np.allclose(np.linalg.det(rotation), 1, atol=1e-5, rtol=0):
        raise ValueError("improper_rotation")
    return {"intrins": intr, "cam_R": rotation.copy(), "cam_t": w2c[:, :3, 3].copy(),
            "width": width, "height": height, "frame_count": n,
            "convention": "world_to_camera_opencv_meters"}
