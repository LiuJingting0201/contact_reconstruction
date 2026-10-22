"""Create the frozen M-S1 report and curves, without accessing E3/E4."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    out = a.run
    completion = json.loads((out / "completion.json").read_text())
    config_files = list(out.glob("resnet50_real_mixed_m_s1*.json"))
    assert len(config_files) == 1, config_files
    config = json.loads(config_files[0].read_text())
    records = [json.loads(x) for x in (out / "evaluations.jsonl").read_text().splitlines()]
    lookup = {r["tag"]: r for r in records}
    steps = [json.loads(x) for x in (out / "steps.jsonl").read_text().splitlines()]
    baseline = lookup["rhd_only_baseline"]["domains"]
    best_step = completion["best_step"]
    chosen = lookup.get(f"dev_{best_step:07d}") if best_step is not None else None
    latest_dev = [r for r in records if r["tag"].startswith("dev_")][-1]
    historical_reference_note = (
        "The historical 32.119 mm / 43.076 mm / 15.712 px values reproduce "
        "on RHD dev, not on the frozen DexYCB subject07 dev. They remain "
        "unchanged as user-requested absolute gates. Improvement estimates "
        "use the RHD-only checkpoint evaluated on the same full DexYCB dev."
    )
    metrics = ("wrist_relative_mpjpe_mm", "fingertip_wrist_relative_error_mm",
               "visible_2d_epe_px", "rotation_aligned_mpjpe_mm",
               "global_orientation_error_deg", "bone_direction_angular_error_deg",
               "projection_collapse_rate", "failure_rate")
    with (out / "curves.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["tag", "domain"] + list(metrics))
        writer.writeheader()
        for record in records:
            for d, m in record["domains"].items():
                writer.writerow({"tag": record["tag"], "domain": d,
                                 **{k: m.get(k) for k in metrics}})
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for col, domain in enumerate(("dexycb", "freihand", "rhd")):
        for row, metric in enumerate(metrics[:2]):
            for prefix, style in (("dev_", "-o"), ("train_diagnostic_", "--x")):
                curve = [r for r in records if r["tag"].startswith(prefix)]
                x = [int(r["tag"].rsplit("_", 1)[-1]) for r in curve]
                y = [r["domains"][domain].get(metric, np.nan) for r in curve]
                axes[row, col].plot(x, y, style, label=prefix.rstrip("_"))
            axes[row, col].axhline(baseline[domain][metric], color="gray", ls=":",
                                  label="RHD-only same dev")
            axes[row, col].set_title(f"{domain}: {metric}")
            axes[row, col].set_xlabel("Updates")
            axes[row, col].legend(fontsize=8)
            axes[row, col].grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "train_dev_geometry_curves.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, metric in zip(axes.flat, ("visible_2d_epe_px", "projection_collapse_rate",
                                     "negative_depth_rate", "nonfinite_rate")):
        for d in ("dexycb", "freihand", "rhd"):
            curve = [r for r in records if r["tag"].startswith("dev_")]
            ax.plot([int(r["tag"].rsplit("_", 1)[-1]) for r in curve],
                    [r["domains"][d].get(metric, np.nan) for r in curve], "-o", label=d)
        ax.set_title(metric)
        ax.set_xlabel("Updates")
        ax.legend()
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "projection_diagnostics.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    finite_steps = [s for s in steps if "loss" in s]
    ax.plot([s["step"] for s in finite_steps], [s["loss"] for s in finite_steps], alpha=.4)
    ax.set(xlabel="Updates", ylabel="Weighted training loss")
    fig.tight_layout()
    fig.savefig(out / "train_loss.png", dpi=160)
    plt.close(fig)
    intervals = {}
    flags = {}
    imbalance = []
    if chosen:
        best = chosen["domains"]["dexycb"]
        old = np.load(out / "rhd_only_baseline_dexycb_predictions.npz")
        new = np.load(out / f"dev_{best_step:07d}_dexycb_predictions.npz")
        assert np.array_equal(old["sample_id"], new["sample_id"])
        groups = np.array([s.split(":")[0] for s in new["sample_id"]])
        unique = sorted(set(groups))
        rng = np.random.default_rng(config["seed"])
        boot = rng.integers(0, len(unique), (2000, len(unique)))
        old3 = np.linalg.norm(old["xyz"] - old["target_xyz"], axis=-1) * 1000
        new3 = np.linalg.norm(new["xyz"] - new["target_xyz"], axis=-1) * 1000
        for name, delta in (
            ("wrist_relative_mpjpe_mm", (new3-old3).mean(1)),
            ("fingertip_wrist_relative_error_mm", (new3-old3)[:, [4, 8, 12, 16, 20]].mean(1)),
        ):
            sums = np.array([delta[groups == group].sum() for group in unique])
            counts = np.array([(groups == group).sum() for group in unique])
            samples = sums[boot].sum(1) / counts[boot].sum(1)
            intervals[name] = {
                "mean_paired_delta": float(delta.mean()),
                "sequence_cluster_bootstrap_95pct_ci": np.quantile(samples, [.025, .975]).tolist(),
                "clusters": len(unique), "replicates": 2000,
            }
        dev_curve = [r for r in records if r["tag"].startswith("dev_") and
                     int(r["tag"].rsplit("_", 1)[-1]) > 0]
        epe = [r["domains"]["dexycb"].get("visible_2d_epe_px", np.nan) for r in dev_curve]
        stable_2d = len(epe) >= 3 and np.isfinite(epe[-3:]).all() and all(
            a >= b for a, b in zip(epe[-3:], epe[-2:]))
        wrist_curve = [r["domains"]["dexycb"].get("wrist_relative_mpjpe_mm", np.nan) for r in dev_curve]
        continuing = len(wrist_curve) >= 3 and np.isfinite(wrist_curve[-3:]).all() and all(
            a > b for a, b in zip(wrist_curve[-3:], wrist_curve[-2:]))
        margin = 1-config["gate"]["minimum_relative_improvement"]
        flags = {
            "wrist_below_historical_by_10pct": best["wrist_relative_mpjpe_mm"] <
                margin*config["gate"]["historical_wrist_mpjpe_mm"],
            "tips_below_historical_by_10pct": best["fingertip_wrist_relative_error_mm"] <
                margin*config["gate"]["historical_fingertip_mm"],
            "wrist_same_dev_paired_ci_below_zero":
                intervals["wrist_relative_mpjpe_mm"]["sequence_cluster_bootstrap_95pct_ci"][1] < 0,
            "tips_same_dev_paired_ci_below_zero":
                intervals["fingertip_wrist_relative_error_mm"]["sequence_cluster_bootstrap_95pct_ci"][1] < 0,
            "2d_improved_or_stably_decreasing": best["visible_2d_epe_px"] <
                margin*config["gate"]["historical_2d_epe_px"] or stable_2d,
            "all_domain_failure_at_most_0_1pct":
                all(m["failure_rate"] <= .001 for m in chosen["domains"].values()),
            "no_collapse_stop": completion["stop_reason"] != "consecutive_projection_collapse",
            "no_nonfinite_stop": not completion["stop_reason"].startswith("nonfinite"),
            "dev_geometry_improved_from_initial": best["wrist_relative_mpjpe_mm"] <
                lookup["dev_0000000"]["domains"]["dexycb"]["wrist_relative_mpjpe_mm"],
            "real_dev_last_three_evaluations_improving": continuing,
        }
        for d in ("freihand", "rhd"):
            for k in metrics[:3]:
                if chosen["domains"][d][k] > baseline[d][k] * 1.25:
                    imbalance.append({"domain": d, "metric": k, "baseline": baseline[d][k],
                                      "best_dev": chosen["domains"][d][k]})
        flags["no_severe_domain_imbalance"] = not imbalance
    flags = {k: bool(v) for k, v in flags.items()}
    passed = bool(flags) and all(flags.values())
    train_collapse_steps = [s for s in steps if s.get("projection_collapse_rate", 0) > 0]
    train_collapse_summary = {
        "affected_updates": len(train_collapse_steps),
        "first_update": train_collapse_steps[0]["step"] if train_collapse_steps else None,
        "last_update": train_collapse_steps[-1]["step"] if train_collapse_steps else None,
        "maximum_batch_rate": max(
            (s["projection_collapse_rate"] for s in train_collapse_steps), default=0),
        "raw_evidence": "steps.jsonl; retained independently of later dev recovery",
    }
    exposure = json.loads((out / "training_sample_exposure.json").read_text())
    coverage = {d: {"eligible_samples": 0, "unique_samples_drawn": 0, "sample_draws": 0}
                for d in ("dexycb", "freihand", "rhd")}
    for line in (out / "splits/train.jsonl").open():
        sample = json.loads(line)
        domain = coverage[sample["dataset"]]
        draws = exposure.get(sample["sample_id"], 0)
        domain["eligible_samples"] += 1
        domain["unique_samples_drawn"] += int(draws > 0)
        domain["sample_draws"] += draws
    assert sum(d["sample_draws"] for d in coverage.values()) == sum(exposure.values())
    for domain in coverage.values():
        domain["coverage_fraction"] = domain["unique_samples_drawn"] / domain["eligible_samples"]
        domain["effective_passes"] = domain["sample_draws"] / domain["eligible_samples"]
    report = {
        "question": "Does real mixed training substantially improve ResNet geometry and 2D on independent real dev?",
        "m_s2_eligible": passed, "gate_flags": flags,
        "paired_same_dev_comparison": intervals, "domain_imbalance": imbalance,
        "domain_imbalance_history": [
            json.loads(line) for line in (out / "domain_imbalance_events.jsonl").read_text().splitlines()
        ] if (out / "domain_imbalance_events.jsonl").exists() else [],
        "best": chosen, "latest_dev": latest_dev,
        "baseline": baseline, "completion": completion,
        "historical_reference_note": historical_reference_note,
        "historical_smoke_failure": config["historical_smoke_failure"],
        "train_collapse_summary": train_collapse_summary,
        "training_pool_coverage": coverage,
        "projection_history": [{"tag": r["tag"], "domains": {
            d: {k: m[k] for k in ("projection_collapse_rate", "negative_depth_rate",
                                  "nonfinite_rate", "projection_span_px", "weak_camera_scale")}
            for d, m in r["domains"].items()}} for r in records],
    }
    (out / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = ["# M-S1 real mixed no-KD result", "",
             f"M-S2 eligible: **{passed}**. No M-S2 execution was performed.",
             f"Stop reason: `{completion['stop_reason']}`. Last update: {completion['last_step']}; "
             f"selected update: {best_step}.", "",
             "## Independent dev results", "",
             "| Domain | Model | Wrist MPJPE mm | Fingertip mm | Visible 2D px | Failure |",
             "|---|---|---:|---:|---:|---:|"]
    for d in ("dexycb", "freihand", "rhd"):
        for name, group in (
            ("RHD-only", baseline),
            ("M-S1 best" if chosen else "M-S1 latest (not selectable)",
             (chosen or latest_dev)["domains"]),
        ):
            if d not in group:
                continue
            m = group[d]
            lines.append(f"| {d} | {name} | {m['wrist_relative_mpjpe_mm']:.3f} | "
                         f"{m['fingertip_wrist_relative_error_mm']:.3f} | "
                         f"{m['visible_2d_epe_px']:.3f} | {m['failure_rate']:.4%} |")
    lines += ["", historical_reference_note,
              "", "## Gate checks", "", *[f"- {k}: {v}" for k, v in flags.items()],
              "", "## Projection-collapse history", "",
              "M-S0 step 100 FreiHAND collapse was 9.375%; that early failure remains in the record. "
              "M-S0 metrics are not accuracy evidence. All M-S1 dev failures, including recoveries, "
              "are retained in `evaluations.jsonl` and `results.json`.",
              f"Training batches with collapse: {train_collapse_summary['affected_updates']}; "
              f"maximum batch rate: {train_collapse_summary['maximum_batch_rate']:.4%}. "
              "Raw per-update evidence remains in `steps.jsonl`.",
              "", "## Training pool coverage", "",
              "| Domain | Eligible | Unique drawn | Sample draws | Effective passes |",
              "|---|---:|---:|---:|---:|",
              *[f"| {d} | {m['eligible_samples']} | {m['unique_samples_drawn']} | "
                f"{m['sample_draws']} | {m['effective_passes']:.3f} |"
                for d, m in coverage.items()],
              "", "All eligible samples were available to the sampler. Draw counts include "
              "attempted batches if training stopped before an optimizer update.",
              "", "## Scope and limitations", "",
              "DexYCB uses subject02 for train, subject07 for full independent dev, and sealed "
              "subjects08/09 for Final-E4. No E4 image/label/model access, E3, KD, B2 tuning, "
              "Dyn-HaMR integration or retargeting occurred. FreiHAND is capture-disjoint with "
              "all four RGB variants grouped together; public subject/sequence IDs are absent, "
              "so subject independence cannot be claimed. Train diagnostic panels select no checkpoints. "
              "RHD-only baseline was evaluated using exactly the same full dev crops and masks. "
              "DexYCB dev contains one independent subject; sequence bootstrap intervals "
              "do not measure uncertainty across subjects. "
              "Rotation alignment is wrist-anchored proper rotation, without scale fitting. "
              "Per-finger errors exclude the common wrist. The aggregate is an unweighted domain mean.",
              "", "Severe domain regression requires sampler diagnosis and a separately registered "
              "follow-up; a favorable aggregate cannot override it.",
              "", "## Checkpoint SHA256", "",
              *[f"- `{name}`: `{digest}`" for name, digest in completion["checkpoint_hashes"].items()],
              "", "See `splits/`, the copied JSON configuration, `curves.csv`, and PNG curves."]
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"m_s2_eligible": passed, "best_step": best_step, "gate_flags": flags}), flush=True)


if __name__ == "__main__":
    main()
