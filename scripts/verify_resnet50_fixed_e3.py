"""Independently verify a completed fixed-E3 run without inference or optimization."""
import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_pose_preserving_e2 import one, stats
from real_e2_data import sha256

ROOT = Path(__file__).resolve().parents[1]


def close(actual, expected, label, atol=1e-8):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=1e-7,
                               err_msg=label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    result = json.loads((run / "results.json").read_text())
    config = json.loads((run / "frozen_b2_config.json").read_text())
    source = ROOT / "artifacts/dex_real_e2_matrix_20261020_02"
    rows = {"raw": [], "B2": []}
    checks = []
    for index in range(32):
        folder = run / f"sample_{index:03d}"
        source_folder = source / f"sample_{index:03d}"
        row = json.loads((folder / "sample.json").read_text())
        source_row = json.loads((source_folder / "sample.json").read_text())
        assert row == source_row
        target = np.load(folder / "evaluation.npz")
        cached = np.load(source_folder / "evaluation.npz")
        for key in ("gt", "gt_uv", "gt_camera", "valid", "K", "T", "affine"):
            np.testing.assert_array_equal(target[key], cached[key])
        predictions = np.load(folder / "predictions.npz")
        pseudo = np.stack([predictions[f"{view}_pseudo"] for view in range(2)])
        np.testing.assert_array_equal(target["pseudo"], pseudo)
        sample_metrics = json.loads((folder / "metrics.json").read_text())
        for variant in ("raw", "B2"):
            state = np.load(folder / variant / "state.npz")
            runtime = json.loads((folder / variant / "runtime.json").read_text())
            recomputed = one(state, target, runtime, row, config)
            for key, value in recomputed.items():
                close(value, sample_metrics[variant][key],
                      f"sample {index} {variant} {key}")
            rows[variant].append({
                "sample_id": row["sample_id"], "variant": variant,
                "metrics": recomputed,
            })
        b2 = np.load(folder / "B2/state.npz")
        np.testing.assert_array_equal(b2["rotations"], target["rotations"])
        np.testing.assert_array_equal(b2["betas"], target["betas"])
        assert np.isfinite(b2["xyz"]).all() and (b2["camera"][..., 2] > .01).all()
        checks.append({
            "sample_id": row["sample_id"],
            "rotations_exact": True, "betas_exact": True,
            "calibration_exact": True, "metrics_recomputed": True,
        })
    assert [item["sample_id"] for item in checks] == result["sample_ids"]
    for variant in ("raw", "B2"):
        for key in rows[variant][0]["metrics"]:
            aggregate = stats([item["metrics"][key] for item in rows[variant]])
            for statistic, value in aggregate.items():
                close(value, result["resnet"][variant][key][statistic],
                      f"aggregate {variant} {key} {statistic}")
    checkpoint = Path(result["checkpoint"]["path"])
    assert sha256(checkpoint) == result["checkpoint"]["sha256"]
    assert result["checkpoint"]["selection_used_final_e3"] is False
    hashes = json.loads((run / "sha256.json").read_text())
    for relative, digest in hashes.items():
        assert sha256(run / relative) == digest
    verification = {
        "passed": True,
        "samples": 32,
        "same_ordered_ids": True,
        "source_rows_exact": True,
        "gt_K_T_affine_exact": True,
        "B2_rotations_betas_exact": True,
        "positive_depth_and_finite": True,
        "sample_metrics_recomputed": True,
        "aggregate_metrics_recomputed": True,
        "checkpoint_hash_verified": True,
        "manifest_hashes_verified": True,
        "inference_calls": 0,
        "optimization_calls": 0,
        "rows": checks,
    }
    (run / "verification.json").write_text(
        json.dumps(verification, indent=2, allow_nan=False) + "\n")
    final_hashes = {
        str(path.relative_to(run)): sha256(path) for path in sorted(run.rglob("*"))
        if path.is_file() and path.name != "sha256.json"
    }
    (run / "sha256.json").write_text(
        json.dumps(final_hashes, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in verification.items()
                      if key != "rows"}))


if __name__ == "__main__":
    main()
