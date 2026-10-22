"""Evaluate each camera's own fixed MANO under nominal and real-K interpretations."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np


def stats(x):
    return {"mean": float(np.mean(x)), "median": float(np.median(x)),
            "p95": float(np.percentile(x, 95))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(exist_ok=False)
    shutil.copy2(__file__, a.output/Path(__file__).name)
    rows = []
    for folder in sorted(a.run.glob("sample_*")):
        teacher = np.load(folder/"teacher.npz")
        target = np.load(folder/"evaluation.npz")
        sid = json.loads((folder/"sample.json").read_text())["sample_id"]
        for v, side in enumerate(("left", "right")):
            camera = teacher[f"{v}_xyz"]+teacher[f"{v}_translation"]
            assert camera[:, 2].min() > 0
            h = camera @ target["K"][v].T
            uv = h[:, :2]/h[:, 2:]
            mask = target["valid"][v].astype(bool)
            real = np.linalg.norm(uv-target["gt_uv"][v], axis=-1)[mask].mean()
            nominal = np.linalg.norm(teacher[f"{v}_pseudo"]-target["gt_uv"][v], axis=-1)[mask].mean()
            rows.append({"sample_id": sid, "view": side, "own_real_K_gt2d_px": float(real),
                         "own_nominal_gt2d_px": float(nominal),
                         "delta_nominal_minus_real_K_px": float(nominal-real),
                         "own_real_K_absolute_mpjpe_mm": float(
                             np.linalg.norm(camera-target["gt_camera"][v], axis=-1).mean()*1000),
                         "own_real_K_wrist_depth_error_mm": float(
                             abs(camera[0, 2]-target["gt_camera"][v, 0, 2])*1000)})
    summary = {}
    for side in ("left", "right"):
        selected = [r for r in rows if r["view"] == side]
        summary[side] = {k: stats([r[k] for r in selected])
                         for k in selected[0] if k not in ("sample_id", "view")}
        summary[side]["nominal_improved_count"] = sum(r["delta_nominal_minus_real_K_px"] < 0 for r in selected)
        summary[side]["samples"] = len(selected)
    result = {"rows": rows, "summary": summary, "inference_calls": 0, "optimizer_updates": 0,
              "interpretation": "same per-camera MANO prediction under two camera interpretations",
              "right_comparator": "own right-camera real-K initialization, not transformed left-camera initialization"}
    (a.output/"branch_metrics.json").write_text(json.dumps(result, indent=2)+"\n")
    lines = ["# Per-camera focal interpretation audit", "",
             "Saved predictions only; zero inference or optimizer calls. Each camera uses its own identical MANO prediction for both interpretations.",
             "Unlike the shared E2-0 right-camera metric, this comparison does not change the source camera's MANO prediction.", "",
             "| View | Own real-K EPE mean / median / P95 px | Nominal EPE mean / median / P95 px | Nominal improved |",
             "| --- | --- | --- | --- |"]
    for side, values in summary.items():
        cells = [" / ".join(f"{x:.3f}" for x in values[k].values())
                 for k in ("own_real_K_gt2d_px", "own_nominal_gt2d_px")]
        lines.append(f"| {side} | "+" | ".join(cells)+f" | {values['nominal_improved_count']}/{values['samples']} |")
    lines += ["", "Lower nominal target EPE does not establish virtual depth accuracy or improvement on every sample.", ""]
    (a.output/"results.md").write_text("\n".join(lines))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
