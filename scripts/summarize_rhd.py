"""Descriptive plots and CSV from saved pilot logs; never selects a model."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
settings = json.loads((root/"experiments/rhd_supervised_v2.json").read_text())
parser = argparse.ArgumentParser()
parser.add_argument("--run-id", default=settings["run_id"])
args = parser.parse_args()
out = root/"artifacts"/args.run_id
steps = [json.loads(s) for s in (out/"steps.jsonl").read_text().splitlines()]
evaluations = [json.loads(s) for s in (out/"evaluations.jsonl").read_text().splitlines()]
with (out/"metrics.csv").open("w") as f:
    writer = csv.writer(f)
    writer.writerow(["step","split","mpjpe_mm","visible_crop_error_px"])
    for e in evaluations:
        for split,m in e["splits"].items():
            writer.writerow([e["step"],split,m["mpjpe_mm"],m["visible_crop_error_px"]])
fig,axes = plt.subplots(1,3,figsize=(14,4))
axes[0].plot([s["step"] for s in steps],[s["loss"] for s in steps],linewidth=.8)
axes[0].set(xlabel="Optimizer update",ylabel="Weighted training loss",yscale="log")
for split in ["training","evaluation"]:
    for ax,key in zip(axes[1:],["mpjpe_mm","visible_crop_error_px"]):
        ax.plot([e["step"] for e in evaluations],[e["splits"][split][key] for e in evaluations],
                marker="o",label=split)
axes[1].set(xlabel="Optimizer update",ylabel="Wrist-relative MPJPE (mm)")
axes[2].set(xlabel="Optimizer update",ylabel="Visible crop reprojection error (px)")
for ax in axes:
    ax.grid(alpha=.25)
for ax in axes[1:]:
    ax.legend()
fig.suptitle("RHD subset pilot: 128 train / 32 held-out evaluation; no KD; GT crops")
fig.tight_layout()
fig.savefig(out/"training_curves.png",dpi=160)
plt.close(fig)
first,last = evaluations[0]["splits"]["evaluation"],evaluations[-1]["splits"]["evaluation"]
a,b = np.array(first["per_image_mpjpe_mm"]),np.array(last["per_image_mpjpe_mm"])
summary = {"evaluation_images_with_lower_mpjpe":int((b<a).sum()),
           "evaluation_images":len(a),"initial_median_mpjpe_mm":float(np.median(a)),
           "final_median_mpjpe_mm":float(np.median(b)),"final_p90_mpjpe_mm":float(np.quantile(b,.9)),
           "initial_mpjpe_mm":first["mpjpe_mm"],"final_mpjpe_mm":last["mpjpe_mm"],
           "relative_mpjpe_reduction":1-last["mpjpe_mm"]/first["mpjpe_mm"],
           "initial_visible_crop_error_px":first["visible_crop_error_px"],
           "final_visible_crop_error_px":last["visible_crop_error_px"],
           "caution":"Descriptive pilot statistics only; fixed final checkpoint, no confidence/generalization claim for full dataset or real cameras."}
(out/"descriptive_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
