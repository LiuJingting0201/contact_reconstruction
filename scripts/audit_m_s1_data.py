"""Audit 64 unaugmented samples per domain/split before M-S1 model access."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

import train_resnet50_m_s1 as ms
import train_resnet50_real_mixed as base


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    a = p.parse_args()
    c = json.loads((base.ROOT / a.config).read_text())
    out = base.ROOT / c["audit_dir"]
    out.mkdir(parents=True, exist_ok=False)
    reports = {}
    for split in ("train", "dev"):
        for domain, dataset in ms.datasets(c, split).items():
            if isinstance(dataset, torch.utils.data.Subset):
                dataset.dataset.train = False
            else:
                dataset.train = False
            rows, images, uv_list = [], [], []
            for idx in np.linspace(0, len(dataset)-1, 64).round().astype(int):
                sample = dataset[int(idx)]
                xyz, uv = sample["xyz"].numpy(), sample["uv"].numpy()
                valid = sample["uv_valid"].numpy()
                assert np.isfinite(xyz).all() and np.isfinite(uv).all()
                assert np.max(np.abs(xyz[0])) < 1e-7
                assert valid.sum() > 0
                lengths = np.array([np.linalg.norm(xyz[b]-xyz[a]) for a, b in base.EDGES])
                assert lengths.min() > 0 and lengths.max() < .20
                orient = sample["global_orient"].numpy()
                assert np.allclose(orient @ orient.T, np.eye(3), atol=1e-5)
                assert abs(np.linalg.det(orient) - 1) < 1e-5
                if domain == "dexycb":
                    assert not sample["hand_pose_valid"] and not sample["betas_valid"]
                    row = dataset.rows[int(idx)]
                    raw = np.load(dataset.root / row["label"])
                    absolute = raw["joint_3d"][0]
                    target = raw["joint_2d"][0]
                    intr_path = dataset.root / "calibration/intrinsics" / f"{row['camera_serial']}_640x480.yml"
                    intr = yaml.load(intr_path.read_text(), Loader=yaml.FullLoader)["color"]
                    K = np.array([[intr["fx"], 0, intr["ppx"]],
                                  [0, intr["fy"], intr["ppy"]], [0, 0, 1]])
                    h = absolute @ K.T
                    source_valid = (target >= 0).all(1)
                    project_error = np.linalg.norm(h[:, :2] / h[:, 2:] - target, axis=1)[source_valid]
                    assert project_error.max() < 2
                    expected = absolute - absolute[:1]
                    if row["mano_side"] == "left":
                        expected[:, 0] *= -1
                    assert np.allclose(xyz, expected, atol=1e-7)
                    projection_check = float(project_error.max())
                elif domain == "freihand":
                    assert sample["hand_pose_valid"] and sample["betas_valid"]
                    actual = dataset.indices[int(idx)] % 32560
                    ds = dataset.dataset
                    homogeneous = ds.xyz[actual] @ ds.K[actual].T
                    expected = homogeneous[:, :2] / homogeneous[:, 2:] / 224 - .5
                    projection_check = float(np.max(np.abs(expected-uv)) * 256)
                    assert projection_check < 1e-3
                    assert np.allclose(sample["betas"], ds.mano[actual, 48:58])
                    expected_pose = base.axis_angle_to_matrix(ds.mano[actual, 3:48])
                    assert np.allclose(sample["hand_pose"], expected_pose, atol=1e-6)
                else:
                    assert not sample["global_orient_valid"]
                    assert not sample["hand_pose_valid"] and not sample["betas_valid"]
                    raw = dataset.data
                    expected = raw["xyz_canonical_wrist_relative_m"][idx]
                    assert np.allclose(expected, xyz, atol=1e-6)
                    expected_uv = raw["uv_canonical_crop_px"][idx] / 256 - .5
                    projection_check = float(np.max(np.abs(expected_uv-uv)) * 256)
                    assert projection_check < 1e-3
                rows.append({"sample_id": sample["sample_id"], "valid_joints": int(valid.sum()),
                             "max_projection_disagreement_px": projection_check,
                             "mean_bone_length_m": float(lengths.mean()),
                             "global_orientation_valid": bool(sample["global_orient_valid"]),
                             "hand_pose_valid": bool(sample["hand_pose_valid"]),
                             "betas_valid": bool(sample["betas_valid"])})
                images.append(sample["img"].numpy())
                uv_list.append(uv)
            for page in range(8):
                sl = slice(page*8, page*8+8)
                base.draw_overlay(np.array(images)[sl], np.array(uv_list)[sl], np.array(uv_list)[sl],
                                  out / f"{split}_{domain}_{page}.jpg")
            reports[f"{split}_{domain}"] = {"passed": True, "count": 64, "rows": rows}
    base.write_json(out / "audit.json", {
        "passed": True, "config_sha256": base.sha256(base.ROOT / a.config),
        "freeze_sha256": base.sha256(base.ROOT / c["split_dir"] / "freeze.json"),
        "domains": reports,
        "limitations": [
            "FreiHAND 2D is derived from XYZ/K, not independent annotation.",
            "FreiHAND pose/shape and joint labels are separate supervision; historical "
            "MANO-versus-joints disagreement 2.96mm mean / 8.44mm p95 remains recorded.",
            "DexYCB articulated PCA pose and betas remain masked.",
            "Annotation availability is not a physical occlusion measurement.",
        ],
    })
    print(json.dumps({"passed": True, "audited_samples": 384}), flush=True)


if __name__ == "__main__":
    main()
