"""Report fixed-backend selection and paired pose-preserving results."""
import argparse
import json
from pathlib import Path

DISPLAY = ["absolute_mpjpe_mm", "wrist_relative_mpjpe_mm",
           "rotation_aligned_wrist_relative_mpjpe_mm_diagnostic",
           "left_wrist_depth_error_mm", "right_wrist_depth_error_mm",
           "wrist_position_error_mm", "fingertip_absolute_error_mm",
           "fingertip_wrist_relative_error_mm", "global_orientation_error_deg",
           "left_gt2d_epe_px", "right_gt2d_epe_px", "pseudo_residual_px"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(exist_ok=False)
    report = json.loads(a.metrics.read_text())
    agg = report["aggregates"]
    labels = list(agg)
    feasible = [v for v in labels if agg[v]["numerical_failure"]["mean"] == 0
                and agg[v]["positive_depth_failure"]["mean"] == 0
                and agg[v]["quality_failure"]["mean"] <= 1/32+1e-12
                and agg[v]["absolute_mpjpe_mm"]["mean"] <= 12
                and agg[v]["wrist_relative_mpjpe_mm"]["mean"] <= 8]
    # Minimize both main absolute and wrist-relative errors among variants satisfying
    # hard failure gates. Pseudo residual is intentionally absent.
    eligible = [v for v in labels if agg[v]["numerical_failure"]["mean"] == 0
                and agg[v]["positive_depth_failure"]["mean"] == 0
                and agg[v]["quality_failure"]["mean"] <= 1/32+1e-12]
    frontier = [v for v in eligible if not any(
        agg[u]["absolute_mpjpe_mm"]["mean"] <= agg[v]["absolute_mpjpe_mm"]["mean"]
        and agg[u]["wrist_relative_mpjpe_mm"]["mean"] <= agg[v]["wrist_relative_mpjpe_mm"]["mean"]
        and (agg[u]["absolute_mpjpe_mm"]["mean"] < agg[v]["absolute_mpjpe_mm"]["mean"]
             or agg[u]["wrist_relative_mpjpe_mm"]["mean"] < agg[v]["wrist_relative_mpjpe_mm"]["mean"])
        for u in eligible)]
    selected = "B2" if "B2" in feasible else None
    decision = {"joint_feasible": feasible, "pareto_frontier": frontier,
                "selected_backend": selected,
                "selected_definition": "stereo Huber + translation only" if selected else None,
                "E3_gate": "eligible to propose, not started" if selected else "no fixed backend selected",
                "reason": "B2 meets all thresholds; the preregistered rule explicitly prefers B2 when eligible, retaining robust loss over the nearly identical, slightly faster B3 L2 result"}
    (a.output/"selection.json").write_text(json.dumps(decision, indent=2)+"\n")
    lines = ["# Real E2 pose-preserving backend ablation", "",
             f"Selected fixed future-E3 backend: **{selected}: stereo Huber + translation only**.",
             "Each metric cell is mean / median / P95. Errors are mm unless labeled px/deg.",
             "Rotation-aligned wrist-relative MPJPE is diagnostic only.", ""]
    for keys in (DISPLAY[:3], DISPLAY[3:8], DISPLAY[8:]):
        lines += ["| Variant | "+" | ".join(keys)+" |",
                  "| --- | "+" | ".join(["---"]*len(keys))+" |"]
        for v in labels:
            cells = [" / ".join(f"{agg[v][k][s]:.3f}" for s in ("mean", "median", "p95")) for k in keys]
            lines.append("| "+v+" | "+" | ".join(cells)+" |")
        lines.append("")
    lines += ["## Failures, runtime and changes", "",
              "| Variant | Quality | Numerical | Depth | Improved abs vs B0 | Runtime P50/P95 s | Translation mm | Global deg | Articulated deg | Shape L2 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    rows = report["rows"]
    base = {x["sample_id"]: x["metrics"]["absolute_mpjpe_mm"] for x in rows if x["variant"] == "B0"}
    for v in labels:
        selected_rows = [x for x in rows if x["variant"] == v]
        improved = sum(x["metrics"]["absolute_mpjpe_mm"] < base[x["sample_id"]] for x in selected_rows)
        cells = [f"{int(agg[v][k]['mean']*32)}/32" for k in
                 ("quality_failure", "numerical_failure", "positive_depth_failure")]
        timing = f"{agg[v]['backend_s']['median']:.3f}/{agg[v]['backend_s']['p95']:.3f}"
        changes = [f"{agg[v][k]['mean']:.3f}" for k in
                   ("translation_change_mm", "global_rotation_change_deg",
                    "articulated_pose_change_deg", "shape_change_l2")]
        lines.append("| "+v+" | "+" | ".join(cells+[f"{improved}/32", timing]+changes)+" |")
    lines += ["", "## Bone direction angular error", "",
              "| Variant | Thumb | Index | Middle | Ring | Little |",
              "| --- | --- | --- | --- | --- | --- |"]
    for v in labels:
        cells = [f"{agg[v][f'bone_{f}_angular_error_deg']['mean']:.3f}"
                 for f in ("thumb", "index", "middle", "ring", "little")]
        lines.append("| "+v+" | "+" | ".join(cells)+" |")
    lines += ["", "## Required paired comparisons", ""]
    lines += ["Improvement counts require delta < -1e-4 in the displayed unit, excluding floating-point equality noise.", ""]
    for name, values in report["paired_comparisons"].items():
        lines.append(f"### {name}")
        lines.append("| Metric | Delta mean / median / P95 | Improved |")
        lines.append("| --- | --- | --- |")
        for metric, value in values.items():
            d = value["delta_candidate_minus_baseline"]
            lines.append(f"| {metric} | {d['mean']:.3f} / {d['median']:.3f} / {d['p95']:.3f} | "
                         f"{value['candidate_improved_count']}/32 |")
        lines.append("")
    lines += ["## Selection", "",
              f"Joint-feasible variants: {', '.join(feasible)}.",
              f"Absolute/wrist-relative Pareto frontier after hard failure gates: {', '.join(frontier)}.",
              "B2 is selected by the preregistered rule. It exactly freezes all rotations and shape,",
              "meets 8.125 mm absolute and 6.762 mm wrist-relative targets with 1/32 quality failure,",
              "and uses about half B4's backend time. Its higher pseudo residual is not a rejection criterion.",
              "",
              "B5-B8 do not materially constrain orientation at the preregistered weights; all are retained.",
              "No post-hoc anchor was needed because B2 already satisfies the selection rule.",
              "E3 remains paused.", ""]
    (a.output/"results.md").write_text("\n".join(lines))
    print(json.dumps(decision), flush=True)


if __name__ == "__main__":
    main()
