"""Generate reviewable tables and figures from independent saved real E2 metrics."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = ["left_gt2d_px", "right_gt2d_px", "stereo_gt2d_px",
           "root_relative_mpjpe_mm", "left_absolute_mpjpe_mm", "right_absolute_mpjpe_mm",
           "shared_frame_right_gt_mpjpe_mm", "wrist_position_error_mm",
           "left_wrist_depth_error_mm", "right_wrist_depth_error_mm", "pseudo_residual_px"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(exist_ok=False)
    report = json.loads((a.run/"independent_metrics.json").read_text())
    rows = report["rows"]
    variants = [v for v, values in report["aggregates"].items() if values["samples"]]
    index = {(r["sample_id"], r["variant"]): r for r in rows}
    samples = sorted(set(r["sample_id"] for r in rows))
    require = lambda cond, message: None if cond else (_ for _ in ()).throw(ValueError(message))
    require(all((sid, v) in index for sid in samples for v in variants), "matrix sample mismatch")
    def values(v, key):
        return np.array([index[(sid, v)]["metrics"][key] for sid in samples], float)
    def stats(x):
        return {"mean": float(np.mean(x)), "median": float(np.median(x)),
                "p95": float(np.percentile(x, 95))}
    # Select one global monocular variant by mean absolute error for a conservative
    # retrospective comparator. Never select a different winner per sample.
    best = min(("E2-1", "E2-2"), key=lambda v: np.mean(values(v, "absolute_mpjpe_mm")))
    comparisons = {}
    for v in variants:
        if v in ("E2-0", "E2-1", "E2-2"):
            continue
        comparisons[v] = {}
        for key in ("absolute_mpjpe_mm", "wrist_position_error_mm", "wrist_depth_error_mm",
                    "stereo_gt2d_px", "pseudo_residual_px"):
            delta = values(v, key)-values(best, key)
            seq = {}
            for sequence in sorted(set(r["sequence_id"] for r in rows)):
                mask = np.array([index[(sid, v)]["sequence_id"] == sequence for sid in samples])
                seq[sequence] = {"count": int(mask.sum()), "delta": stats(delta[mask]),
                                 "fraction_improved": float((delta[mask] < 0).mean())}
            comparisons[v][key] = {"delta": stats(delta), "improved_count": int((delta < 0).sum()),
                                   "total": len(delta), "fraction_improved": float((delta < 0).mean()),
                                   "per_sequence": seq}
    # Strict descriptive gate: both captured sequences improve, >half of all
    # samples improve, no numerical/depth/quality failure-count increase, for
    # both unweighted robust and deployable geometric-confidence robust.
    gate = {}
    for v in ("E2-4", "E2-5"):
        improves = {}
        for key in ("absolute_mpjpe_mm", "wrist_position_error_mm"):
            c = comparisons[v][key]
            improves[key] = (c["delta"]["mean"] < 0 and c["fraction_improved"] > .5
                             and all(s["delta"]["mean"] < 0 for s in c["per_sequence"].values()))
        failures = {k: float(values(v, k).mean()-values(best, k).mean())
                    for k in ("numerical_failure", "positive_depth_failure", "quality_failure")}
        gate[v] = {"independent_error_improves_on_majority_and_both_sequences": improves,
                   "failure_rate_delta": failures,
                   "passes_descriptive_gate": any(improves.values()) and all(x <= 0 for x in failures.values())}
    supported = all(gate[v]["passes_descriptive_gate"] for v in gate)
    result = {"best_monocular": best, "selection": "lowest mean absolute MPJPE; retrospective global comparator",
              "paired_comparisons": comparisons, "descriptive_gate": gate,
              "supports_claim_on_this_subset": supported,
              "E3": "not started",
              "inference_limit": "2 correlated sequences from one subject and one grasped object; no population significance claim"}
    (a.output/"paired_analysis.json").write_text(json.dumps(result, indent=2)+"\n")
    lines = ["# Official DexYCB real E2 results", "",
             f"{len(samples)} synchronized times. Best monocular by mean absolute MPJPE: **{best}**.",
             "Each cell is mean / median / P95. Pixel metrics are full-image pixels; 3D errors are millimeters.", ""]
    for metrics in (METRICS[:3], METRICS[3:7], METRICS[7:]):
        lines += ["| Variant | "+" | ".join(metrics)+" |",
                  "| --- | "+" | ".join(["---"]*len(metrics))+" |"]
        for v in variants:
            cells = [" / ".join(f"{z:.3f}" for z in stats(values(v, k)).values()) for k in metrics]
            lines.append("| "+v+" | "+" | ".join(cells)+" |")
        lines.append("")
    lines += ["| Variant | Numerical failures | Positive-depth failures | Quality failures | Backend seconds mean / P50 / P95 | Peak allocated GiB |",
              "| --- | --- | --- | --- | --- | --- |"]
    for v in variants:
        failures = [f"{int(values(v,k).sum())}/{len(samples)}" for k in
                    ("numerical_failure", "positive_depth_failure", "quality_failure")]
        timing = " / ".join(f"{z:.3f}" for z in stats(values(v, "backend_s")).values())
        lines.append("| "+v+" | "+" | ".join(failures)+f" | {timing} | {values(v,'peak_allocated_gpu_bytes').max()/2**30:.3f} |")
    lines += ["", "Backend time is serial per-sample offline optimization. It excludes frontend, I/O and reload, and is not FPS.",
              "Peak allocation includes the resident HaMeR model. CUDA_VISIBLE_DEVICES=1 was used.", "",
              "## Optimization budgets and parameter changes", "",
              "Parameter-change cells are mean / median / P95; per-joint angles and each beta/translation component are in independent_metrics.json.", "",
              "| Variant | Translation change mm | Global orientation deg | Mean articulated rotation deg | Beta L2 change | Updates | Objective forwards | Closures |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for v in variants:
        deltas = [" / ".join(f"{z:.3f}" for z in stats(values(v, k)).values())
                  for k in ("translation_change_mm", "global_orientation_change_deg",
                            "articulated_rotation_change_deg", "beta_l2_change")]
        counts = ["/".join(map(str, sorted(set(values(v, k).astype(int).tolist()))))
                  for k in ("optimizer_steps", "forward_objective_calls", "closure_calls")]
        lines.append("| "+v+" | "+" | ".join(deltas+counts)+" |")
    lines += ["", "## Nominal pseudo targets versus real-K shared initialization", "",
              "Both use the same MANO inference. The nominal target in each camera is that camera's own teacher; the shared real-K initialization uses the left teacher transformed into both cameras.", "",
              "| View | Nominal pseudo GT EPE mean / median / P95 px | Shared real-K initialization EPE px | Fraction with lower nominal-target GT EPE |",
              "| --- | --- | --- | --- |"]
    for side in ("left", "right"):
        nominal = values("E2-0", f"{side}_pseudo_gt2d_px")
        real = values("E2-0", f"{side}_gt2d_px")
        cells = [" / ".join(f"{z:.3f}" for z in stats(x).values()) for x in (nominal, real)]
        lines.append("| "+side+" | "+" | ".join(cells)+f" | {(nominal < real).sum()}/{len(samples)} |")
    lines += ["", "## Paired comparison", "",
              "| Variant vs best monocular | Absolute MPJPE delta mean | Absolute improved fraction | Wrist position delta mean | Wrist improved fraction |",
              "| --- | --- | --- | --- | --- |"]
    for v, c in comparisons.items():
        absolute, wrist = c["absolute_mpjpe_mm"], c["wrist_position_error_mm"]
        lines.append(f"| {v} | {absolute['delta']['mean']:.3f} | {absolute['improved_count']}/{len(samples)} | "
                     f"{wrist['delta']['mean']:.3f} | {wrist['improved_count']}/{len(samples)} |")
    lines += ["", "## Failure/degeneration examples", "",
              "Worst absolute-error increases relative to the best monocular variant (no samples removed):", ""]
    for v in ("E2-3", "E2-4", "E2-5", "E2-8"):
        delta = values(v, "absolute_mpjpe_mm")-values(best, "absolute_mpjpe_mm")
        lines.append(f"- {v}: "+", ".join(f"{samples[j]} ({delta[j]:+.2f} mm)" for j in np.argsort(delta)[-3:][::-1]))
    lines += ["", "## Interpretation limits", "",
              "- Dataset joint_2d is geometrically consistent with dataset-fitted joint_3d; it is not established as independent manual 2D measurement.",
              "- Dataset GT is independent of this teacher but has MANO fitting/annotation bias; pretraining overlap unknown.",
              "- Both sequences involve bowl grasping by one subject. Full pose/scene diversity and population significance are not established.",
              "- Sample selection requires successful detections in both views and valid GT. Detector failures are retained in preparation logs, outside backend-failure denominators.",
              "- E2-5 is an epipolar geometric heuristic, not learned confidence or a complete occlusion estimator.",
              "- E2-6 unavailable: official oracle joint visibility not established.",
              "- Same official frame correspondence does not prove zero physical synchronization offset.",
              f"- Descriptive real-stereo gate on this subset: **{'passes' if supported else 'does not pass'}**; E3 has not started.",
              ""]
    (a.output/"results.md").write_text("\n".join(lines))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    for ax, key, title in zip(axes, ("absolute_mpjpe_mm", "wrist_depth_error_mm", "pseudo_residual_px"),
                              ("Absolute MPJPE (mm)", "Left wrist depth error (mm)", "Pseudo fitting residual (px)")):
        arrays = [values(v, key) for v in variants]
        ax.boxplot(arrays, labels=variants, showmeans=True, showfliers=True)
        ax.tick_params(axis="x", rotation=45)
        ax.set_title(title)
        ax.grid(axis="y", alpha=.25)
    fig.suptitle("Official DexYCB: 32 synchronized times, detector crops, fixed 800-update budget")
    fig.tight_layout()
    fig.savefig(a.output/"distributions.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"best_monocular": best, "supports_claim_on_this_subset": supported,
                      "report": str(a.output/"results.md")}), flush=True)


if __name__ == "__main__":
    main()
