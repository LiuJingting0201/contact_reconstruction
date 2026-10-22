"""Static audit figure from saved results; no checkpoint selection."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
base = ROOT/"artifacts"
summary = json.loads((base/"debug_experiment_summary.json").read_text())
fig,axes = plt.subplots(1,3,figsize=(16,4.5))
for run,label in [("rhd_v4_deterministic_a","Positive camera"),
                  ("rhd_v5_deterministic","Positive camera + 10x 2D")]:
    evaluations = [json.loads(x) for x in (base/run/"evaluations.jsonl").read_text().splitlines()]
    axes[0].plot([e["step"] for e in evaluations],
                 [e["splits"]["evaluation"]["visible_crop_error_px"] for e in evaluations],
                 marker="o",label=label)
axes[0].axhline(summary["teacher"]["pooled_visible_crop_error_px"],ls="--",color="gray",label="Pretrained HaMeR")
axes[0].set(xlabel="Optimizer update",ylabel="Development visible 2D error (crop px)",title="Deterministic 200-step comparison")
evaluations = [json.loads(x) for x in (base/"rhd_supervised_v5_long/evaluations.jsonl").read_text().splitlines()]
for split in ["training","evaluation"]:
    axes[1].plot([e["step"] for e in evaluations],
                 [e["splits"][split]["visible_crop_error_px"] for e in evaluations],marker="o",label=split)
axes[1].set(xlabel="Optimizer update",ylabel="Visible 2D error (crop px)",title="Longer training on only 128 images")
teacher = summary["teacher"]
for key,label in [("visible_crop_error_px","2D error (px)"),("wrist_depth_abs_error_mm","Wrist depth error (mm)")]:
    axes[2].plot(["Real-K init","Pseudo-2D fit","GT-2D oracle"],
                 [teacher[s][key] for s in ["initial","pseudo_fit","gt_oracle"]],marker="o",label=label)
axes[2].set(ylabel="Mean per-image error (units in legend)",title="Teacher + translation-only real-K fit")
for ax in axes:
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
fig.suptitle("RHD exploratory development controls | GT crops, right hands | no stereo")
fig.tight_layout()
fig.savefig(base/"debug_comparison.png",dpi=180)
