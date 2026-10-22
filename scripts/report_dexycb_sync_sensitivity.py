"""Paired real-frame sensitivity report; cached synchronous baseline, no optimization."""
import argparse
import json
from pathlib import Path
import numpy as np


def stats(values):
    x = np.asarray(values, float)
    return {"mean": float(x.mean()), "median": float(np.median(x)),
            "p95": float(np.percentile(x, 95))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-run", type=Path, required=True)
    parser.add_argument("--sync-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    a.output.mkdir(exist_ok=False)
    main_rows = json.loads((a.main_run/"independent_metrics.json").read_text())["rows"]
    sync_rows = json.loads((a.sync_run/"independent_metrics.json").read_text())["rows"]
    index = {(r["sample_id"], r["variant"]): r for r in main_rows+sync_rows}
    ids = json.loads((a.sync_run/"selection.json").read_text())["selected"]
    variants = ("E2-4", "E2-S1", "E2-S2")
    assert ids and all((sid, v) in index for sid in ids for v in variants)
    metrics = ("left_gt2d_px", "right_gt2d_px", "absolute_mpjpe_mm",
               "right_absolute_mpjpe_mm", "root_relative_mpjpe_mm",
               "wrist_position_error_mm", "left_wrist_depth_error_mm",
               "right_wrist_depth_error_mm", "pseudo_residual_px",
               "numerical_failure", "positive_depth_failure", "quality_failure", "backend_s")
    summary = {}
    comparisons = {}
    for v in variants:
        summary[v] = {}
        comparisons[v] = {}
        for metric in metrics:
            values = np.array([index[(sid, v)]["metrics"][metric] for sid in ids], float)
            base = np.array([index[(sid, "E2-4")]["metrics"][metric] for sid in ids], float)
            summary[v][metric] = stats(values)
            seq_stats = {}
            for seq in sorted({index[(sid, v)]["sequence_id"] for sid in ids}):
                mask = np.array([index[(sid, v)]["sequence_id"] == seq for sid in ids])
                seq_stats[seq] = stats((values-base)[mask])
            comparisons[v][metric] = {
                "delta_offset_minus_synchronous": stats(values-base),
                "fraction_worse": float((values > base).mean()),
                "per_sequence_delta": seq_stats,
                "per_sample": [{"sample_id": sid, "offset": float(x), "synchronous": float(b),
                                "delta": float(x-b)} for sid, x, b in zip(ids, values, base)]}
    result = {"samples": len(ids), "sample_ids": ids, "aggregates": summary,
              "paired_comparisons": comparisons, "S0_reused": True,
              "reference": "synchronous original GT; only right RGB and pseudo target shifted",
              "time": "one frame is nominally 33.333 ms; physical per-camera timestamps unavailable",
              "scope": "common detector-available neighbors; correlated frames, descriptive statistics only"}
    (a.output/"paired_sync_analysis.json").write_text(json.dumps(result, indent=2)+"\n")
    lines = ["# Real DexYCB synchronization sensitivity", "",
             f"{len(ids)} common times; S0 reuses main E2-4, without rerunning optimization.",
             "Each cell is mean / median / P95. S1 shifts right frame −1; S2 shifts +1.",
             "GT and initialization stay at the original synchronized time.", "",
             "| Metric | S0 (cached E2-4) | S1 (−1) | S2 (+1) |",
             "| --- | --- | --- | --- |"]
    for metric in metrics:
        cells = [" / ".join(f"{x:.3f}" for x in summary[v][metric].values()) for v in variants]
        lines.append("| "+metric+" | "+" | ".join(cells)+" |")
    lines += ["", "| Offset | Abs MPJPE delta mean / median / P95 (mm) | Fraction worse |",
              "| --- | --- | --- |"]
    for v in variants[1:]:
        c = comparisons[v]["absolute_mpjpe_mm"]
        cells = " / ".join(f"{x:.3f}" for x in c["delta_offset_minus_synchronous"].values())
        lines.append(f"| {v} | {cells} | {c['fraction_worse']:.1%} |")
    lines += ["", "Pseudo residual uses each offset's actual optimization targets and therefore is not",
              "a measure of synchronous physical accuracy. The original independent GT is unchanged.",
              "Nominal ±33.333 ms does not establish measured camera latency. Small offsets can",
              "occasionally improve a noisy prediction; do not assume monotonic degradation.", ""]
    (a.output/"results.md").write_text("\n".join(lines))
    print(json.dumps({"samples": len(ids), "output": str(a.output)}), flush=True)


if __name__ == "__main__":
    main()
