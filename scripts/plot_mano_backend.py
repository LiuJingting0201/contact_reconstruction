"""Plot archived E1 backend outcomes and optimizer trajectories."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
directory = root / "artifacts/remote/e1_mano_run02"
if not directory.exists():
    directory = root / "artifacts/e1_mano_run02"
report = json.loads((directory / "results.json").read_text())["results"]
names = list(report)
labels = ["Initial", "T", "T + orient", "T + pose", "T + pose + shape"]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, key, title in zip(axes.flat, [
    "gt2d_crop_px", "root_relative_mpjpe_mm", "absolute_mpjpe_mm",
    "wrist_depth_error_mm", "quality_failure_rate"],
    ["Independent visible GT 2D (crop px)", "Root-relative MPJPE (mm)",
     "Absolute MPJPE (mm)", "Absolute wrist depth error (mm)",
     "Diagnostic quality failure rate"]):
    values = [report[n][key] if key.endswith("rate") else report[n]["mean"][key] for n in names]
    ax.bar(labels, values, color=["#64748b", "#2563eb", "#059669", "#7c3aed", "#d97706"])
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=22)
    ax.set_ylim(0, max(values)*1.2)
for name, label in zip(names[1:], labels[1:]):
    trace = np.load(directory / f"{name}_trajectory.npz")
    axes[1, 2].plot(trace["objective"].mean(1), label=label)
axes[1, 2].set_yscale("log")
axes[1, 2].set_title("Mean fitting objective (pseudo 2D + priors)")
axes[1, 2].set_xlabel("Adam update")
axes[1, 2].legend()
fig.suptitle("E1: fixed pretrained HaMeR, 32 RHD development crops, 800 updates\n"
             "GT used only for evaluation; monocular evidence, not calibrated stereo validation")
fig.tight_layout()
fig.savefig(root / "artifacts/e1_backend_comparison.png", dpi=160)
print(root / "artifacts/e1_backend_comparison.png")
