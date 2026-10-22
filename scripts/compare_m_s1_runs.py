"""Summarize the completed M-S1 run and its single registered correction."""
import argparse
import json
from pathlib import Path

import numpy as np


DOMAINS = ("dexycb", "freihand", "rhd")
METRICS = ("wrist_relative_mpjpe_mm", "fingertip_wrist_relative_error_mm",
           "visible_2d_epe_px")


def read(path):
    return json.loads(path.read_text())


def paired_2d(run, report, seed):
    """Post-hoc descriptive CI; does not change the registered selection/gates."""
    step = report["completion"]["best_step"]
    with np.load(run / "rhd_only_baseline_dexycb_predictions.npz") as old, np.load(
        run / f"dev_{step:07d}_dexycb_predictions.npz"
    ) as new:
        assert np.array_equal(old["sample_id"], new["sample_id"])
        assert np.array_equal(old["valid"], new["valid"])
        assert np.array_equal(old["target_uv"], new["target_uv"])
        valid = old["valid"]
        error_old = np.linalg.norm(old["uv"] - old["target_uv"], axis=-1) * 256
        error_new = np.linalg.norm(new["uv"] - new["target_uv"], axis=-1) * 256
        delta = np.where(valid, error_new - error_old, 0).sum(1)
        count = valid.sum(1)
        groups = np.array([s.split(":")[0] for s in old["sample_id"]])
    unique = sorted(set(groups))
    sums = np.array([delta[groups == group].sum() for group in unique])
    counts = np.array([count[groups == group].sum() for group in unique])
    indices = np.random.default_rng(seed).integers(0, len(unique), (2000, len(unique)))
    changes = sums[indices].sum(1) / counts[indices].sum(1)
    return {
        "mean_paired_delta_px": float(delta.sum() / count.sum()),
        "sequence_cluster_bootstrap_95pct_ci": np.quantile(changes, [.025, .975]).tolist(),
        "clusters": len(unique),
        "replicates": 2000,
        "interpretation": "Descriptive, post-selection; one dev subject, no subject-level CI.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--correction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {"source": args.source, "correction": args.correction}
    reports = {name: read(path / "results.json") for name, path in paths.items()}
    configs = {}
    for name, path in paths.items():
        verification = read(path / "verification.json")
        assert verification["passed"]
        assert reports[name]["completion"]["best_step"] == verification["selected_step"]
        assert reports[name]["best"] is not None
        config_files = list(path.glob("resnet50_real_mixed_m_s1*.json"))
        assert len(config_files) == 1
        configs[name] = read(config_files[0])
    a, b = configs["source"], configs["correction"]
    for key in ("seed", "model", "datasets", "final_manifest_roots", "training",
                "augmentation", "positive_camera", "losses", "label_policy",
                "baseline_checkpoint", "baseline_sha256", "gate", "scope"):
        assert a[key] == b[key], key
    assert b["sampler_correction"]["source_run"] == args.source.name
    assert b["sampler_correction"]["maximum_correction_runs"] == 1
    source_splits = read(args.source / "splits/freeze.json")
    correction_splits = read(args.correction / "splits/freeze.json")
    assert source_splits["split_files"] == correction_splits["split_files"]
    for domain in DOMAINS:
        for metric in METRICS:
            assert np.isclose(reports["source"]["baseline"][domain][metric],
                              reports["correction"]["baseline"][domain][metric],
                              atol=1e-5, rtol=1e-5), (domain, metric)
    baseline = reports["source"]["baseline"]
    changes = {
        name: {
            domain: {
                metric: 100 * (1 - report["best"]["domains"][domain][metric]
                               / baseline[domain][metric])
                for metric in METRICS
            } for domain in DOMAINS
        } for name, report in reports.items()
    }
    epe_ci = {name: paired_2d(paths[name], report, a["seed"])
              for name, report in reports.items()}
    real_dev_improved = {
        name: all(changes[name]["dexycb"][metric] > 0 for metric in METRICS)
        for name in reports
    }
    diagnostic_gaps = {}
    for name, report in reports.items():
        step = report["completion"]["best_step"]
        records = [json.loads(line) for line in
                   (paths[name] / "evaluations.jsonl").read_text().splitlines()]
        train = next(r for r in records if r["tag"] == f"train_diagnostic_{step:07d}")
        diagnostic_gaps[name] = {
            domain: {
                metric: {
                    "train_diagnostic": train["domains"][domain][metric],
                    "full_dev": report["best"]["domains"][domain][metric],
                    "dev_minus_train_diagnostic": report["best"]["domains"][domain][metric]
                    - train["domains"][domain][metric],
                } for metric in METRICS
            } for domain in DOMAINS
        }
    result = {
        "question": "Does real mixed training improve geometry and 2D on independent real dev?",
        "all_three_real_dev_errors_lower_than_same_dev_baseline": real_dev_improved,
        "split_files_identical": True,
        "training_settings_identical_except_sampler": True,
        "samplers": {name: config["batch_composition"] for name, config in configs.items()},
        "relative_improvement_percent": changes,
        "selected_checkpoint_train_dev_diagnostic_gaps": diagnostic_gaps,
        "descriptive_dexycb_2d_paired_comparison": epe_ci,
        "runs": {name: {
            "path": str(paths[name]),
            "completion": report["completion"],
            "m_s2_eligible": report["m_s2_eligible"],
            "gate_flags": report["gate_flags"],
            "domain_imbalance": report["domain_imbalance"],
            "train_collapse_summary": report["train_collapse_summary"],
        } for name, report in reports.items()},
        "further_training_started": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# M-S1 final comparison", "",
        "Both runs completed and passed saved-artifact verification. The second run is the "
        "single registered sampler correction. Each checkpoint was selected using its own "
        "full independent DexYCB dev evaluations and the same safety rule.", "",
        "## Independent dev results", "",
        "| Domain | Model | Wrist MPJPE mm | Fingertip mm | Visible 2D px | Failure |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for domain in DOMAINS:
        groups = [("RHD-only baseline", baseline[domain])]
        groups += [(name, report["best"]["domains"][domain]) for name, report in reports.items()]
        for name, m in groups:
            lines.append(f"| {domain} | {name} | "
                         f"{m[METRICS[0]]:.3f} | {m[METRICS[1]]:.3f} | "
                         f"{m[METRICS[2]]:.3f} | {m['failure_rate']:.4%} |")
    lines += ["", reports["source"]["historical_reference_note"], "",
              "## Per-run conclusions", ""]
    for name, report in reports.items():
        completion = report["completion"]
        lines += [
            f"### {name}", "",
            f"Run: `{paths[name].name}`. Full metrics and curves: "
            f"[results.md](../{paths[name].name}/results.md).",
            f"Stop: `{completion['stop_reason']}` at {completion['last_step']} updates; "
            f"best-dev: {completion['best_step']}. M-S2 eligible: **{report['m_s2_eligible']}**.",
            "All three DexYCB dev errors are lower than the same-dev baseline: "
            f"**{real_dev_improved[name]}**.",
            "DexYCB improvement over the same-dev RHD-only baseline: " + ", ".join(
                f"{label} {changes[name]['dexycb'][metric]:.2f}%"
                for label, metric in zip(("wrist", "fingertip", "2D"), METRICS)) + ".",
            "Failed gates: " + ", ".join(k for k, v in report["gate_flags"].items() if not v) + ".",
            "Severe regression: " + (", ".join(
                f"{item['domain']}/{item['metric']}" for item in report["domain_imbalance"])
                or "none") + ".",
            f"Training collapse history: `{json.dumps(report['train_collapse_summary'])}`.",
            f"Descriptive paired 2D change: `{json.dumps(epe_ci[name])}`.", "",
            *[f"- {ckpt} SHA256: `{digest}`"
              for ckpt, digest in completion["checkpoint_hashes"].items()], "",
        ]
    lines += [
        "## Train/dev diagnostic gap at selected checkpoints", "",
        "| Run | DexYCB train diagnostic wrist mm | Full DexYCB dev wrist mm | Gap mm |",
        "|---|---:|---:|---:|",
        *[f"| {name} | {values['dexycb'][METRICS[0]]['train_diagnostic']:.3f} | "
          f"{values['dexycb'][METRICS[0]]['full_dev']:.3f} | "
          f"{values['dexycb'][METRICS[0]]['dev_minus_train_diagnostic']:.3f} |"
          for name, values in diagnostic_gaps.items()], "",
        "The training diagnostic uses 256 fixed, unaugmented samples per domain and "
        "never selects checkpoints. It is a diagnostic panel, not a full-training-set "
        "estimate. Independent dev improvement does not eliminate the remaining "
        "train/dev gap or establish generalization beyond this dev subject.", "",
        "## Interpretation and limits", "",
        "Geometry confidence intervals are reported in each run's results.json. The additional "
        "2D intervals above are descriptive and were computed after checkpoint selection. "
        "All intervals cluster by sequence within one development subject; they do not "
        "establish cross-subject generalization or adjust for checkpoint selection.",
        "The full training pools were eligible; finite update budgets do not guarantee every "
        "sample was drawn. Actual exposure is reported per domain in each run.",
        "The correction changes batch composition from DexYCB/FreiHAND/RHD 16/10/4 to 12/10/8. "
        "It uses fresh initialization with unchanged losses, optimizer and schedule. "
        "An NPZ metadata caching optimization was applied before the correction; "
        "the original training script snapshot is retained.",
        "No third correction or M-S2 was started. Final-E4 remains sealed. "
        "No E3, KD, B2 tuning, Dyn-HaMR integration or retargeting was run.", "",
    ]
    (args.output / "comparison.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({name: report["m_s2_eligible"] for name, report in reports.items()}))


if __name__ == "__main__":
    main()
