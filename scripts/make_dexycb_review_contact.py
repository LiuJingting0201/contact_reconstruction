"""Enlarge persisted prediction overlays for visual audit; GT crops are audit-only."""
import argparse
import json
from pathlib import Path
import shutil
import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(exist_ok=False)
    shutil.copy2(__file__, a.output/Path(__file__).name)
    rows = json.loads((a.run/"independent_metrics.json").read_text())["rows"]
    best = json.loads((a.report/"paired_analysis.json").read_text())["best_monocular"]
    folders = sorted(a.run.glob("sample_*"))
    by_id = {json.loads((f/"sample.json").read_text())["sample_id"]: f for f in folders}
    index = {(r["sample_id"], r["variant"]): r["metrics"] for r in rows}

    def tile(folder, variant):
        image = cv2.imread(str(folder/variant/"overlay.jpg"))
        uv = np.load(folder/"evaluation.npz")["gt_uv"]
        views = []
        for v in range(2):
            x0, y0 = np.floor(uv[v].min(0)-25).astype(int)
            x1, y1 = np.ceil(uv[v].max(0)+25).astype(int)
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(640, x1), min(480, y1)
            crop = image[y0:y1, v*640+x0:v*640+x1]
            side = max(crop.shape[:2])
            crop = cv2.copyMakeBorder(crop, 0, side-crop.shape[0], 0, side-crop.shape[1],
                                     cv2.BORDER_CONSTANT)
            views.append(cv2.resize(crop, (320, 320)))
        strip = np.zeros((350, 640, 3), np.uint8)
        strip[30:] = np.concatenate(views, axis=1)
        label = f"{folder.name} {variant}: GT green / pseudo blue / fit red"
        cv2.putText(strip, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1)
        return strip

    selected = [0, 5, 10, 15, 16, 21, 26, 31]
    for part in range(2):
        figure = np.concatenate([tile(folders[i], "E2-4") for i in selected[part*4:(part+1)*4]], axis=0)
        cv2.imwrite(str(a.output/f"eight_pairs_{part+1}.jpg"), figure)
    examples, comparisons = [], []
    for variant in ("E2-3", "E2-4", "E2-5", "E2-8"):
        sid = max(by_id, key=lambda s: index[s, variant]["absolute_mpjpe_mm"]
                  - index[s, best]["absolute_mpjpe_mm"])
        examples.append({"variant": variant, "sample_id": sid, "folder": by_id[sid].name,
                         "absolute_delta_mm": index[sid, variant]["absolute_mpjpe_mm"]
                         - index[sid, best]["absolute_mpjpe_mm"]})
        comparisons.append(np.concatenate([tile(by_id[sid], v) for v in ("E2-0", best, variant)], axis=1))
    cv2.imwrite(str(a.output/"worst_comparisons.jpg"), np.concatenate(comparisons, axis=0))
    (a.output/"selection.json").write_text(json.dumps({
        "eight_pair_indices": selected, "worst_examples": examples,
        "crop": "GT extent +25px for visualization audit only; no inference or optimization uses this crop"
    }, indent=2)+"\n")
    print(json.dumps({"output": str(a.output), "worst_examples": examples}), flush=True)


if __name__ == "__main__":
    main()
