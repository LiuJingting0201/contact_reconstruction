"""Render a deterministic train-split prediction overlay from a frozen checkpoint."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from train_resnet50_rhd_full import ROOT, RHDDataset, configure_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    data = RHDDataset(ROOT / config["dataset"] / "train.npz", False, config)
    model, _, _ = configure_model(config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    model.to(device).eval()
    indices = np.linspace(0, len(data) - 1, 32, dtype=int)
    tiles, rows = [], []
    with torch.no_grad():
        for index in indices:
            sample = data[int(index)]
            output = model.forward_step({"img": sample["img"][None].to(device)})
            uv = output["pred_keypoints_2d"][0].cpu().numpy()
            target = sample["uv"].numpy()
            visible = sample["visible"].numpy()
            camera = output["pred_cam"][0].cpu().numpy()
            span = float(np.ptp(uv, axis=0).max() * 256)
            rgb = sample["img"].numpy().transpose(1, 2, 0)
            rgb = np.clip((rgb * np.asarray([.229, .224, .225]) +
                           np.asarray([.485, .456, .406])) * 255, 0, 255).astype(np.uint8)
            for points, color in ((target, (0, 255, 0)), (uv, (255, 0, 0))):
                for x, y in (points + .5) * 256:
                    cv2.circle(rgb, (round(float(x)), round(float(y))), 2, color, -1)
            tiles.append(rgb)
            rows.append({
                "sample_id": sample["sample_id"],
                "visible_2d_epe_px": float(
                    np.linalg.norm(uv - target, axis=-1)[visible].mean() * 256),
                "projection_span_px": span,
                "weak_camera_scale": float(camera[0]),
                "finite": bool(np.isfinite(uv).all() and np.isfinite(camera).all()),
            })
    contact = np.concatenate([
        np.concatenate(tiles[start:start + 8], axis=1) for start in range(0, 32, 8)
    ], axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), cv2.cvtColor(contact, cv2.COLOR_RGB2BGR))
    summary = {
        "checkpoint": str(args.checkpoint),
        "count": len(rows),
        "visible_2d_epe_px_mean": float(np.mean([row["visible_2d_epe_px"] for row in rows])),
        "projection_span_px_min": float(min(row["projection_span_px"] for row in rows)),
        "weak_camera_scale_min": float(min(row["weak_camera_scale"] for row in rows)),
        "all_finite": all(row["finite"] for row in rows),
        "rows": rows,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}))


if __name__ == "__main__":
    main()
