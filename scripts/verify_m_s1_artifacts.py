"""Independently verify saved M-S1 artifacts; never construct or call a model."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(actual, expected, name):
    if not np.isclose(actual, expected, rtol=1e-5, atol=1e-5):
        raise AssertionError(f"{name}: recorded={actual}, recomputed={expected}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    out = args.run
    config_files = list(out.glob("resnet50_real_mixed_m_s1*.json"))
    assert len(config_files) == 1, config_files
    config_path = config_files[0]
    config = json.loads(config_path.read_text())
    provenance = json.loads((out / "provenance.json").read_text())
    completion = json.loads((out / "completion.json").read_text())
    report = json.loads((out / "results.json").read_text())
    frozen = json.loads((out / "splits/freeze.json").read_text())
    assert sha256(config_path) == provenance["config_sha256"] == frozen["config_sha256"]
    assert sha256(out / "train_resnet50_m_s1.py") == provenance["script_sha256"]
    assert sha256(out / "train_resnet50_real_mixed.py") == provenance["base_script_sha256"]
    domains = ("dexycb", "freihand", "rhd")
    expected_ids = {d: [] for d in domains}
    subjects, sequences = {}, {}
    for split in ("train", "dev", "final_e4"):
        path = out / "splits" / f"{split}.jsonl"
        assert sha256(path) == frozen["split_files"][path.name]["sha256"]
        rows = [json.loads(line) for line in path.open()]
        assert len(rows) == frozen["split_files"][path.name]["count"]
        subjects[split] = {r["subject"] for r in rows if r["dataset"] == "dexycb"}
        sequences[split] = {r["sequence"] for r in rows if r["dataset"] == "dexycb"}
        for row in rows:
            digest = row.pop("sha256")
            assert hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest() == digest
            if split == "dev":
                expected_ids[row["dataset"]].append(row["sample_id"])
    for a, b in (("train", "dev"), ("train", "final_e4"), ("dev", "final_e4")):
        assert subjects[a].isdisjoint(subjects[b])
        assert sequences[a].isdisjoint(sequences[b])
    assert subjects["train"] == {"20200813-subject-02"}
    assert subjects["dev"] == {"20200928-subject-07"}
    assert subjects["final_e4"] == {"20201002-subject-08", "20201015-subject-09"}
    for name, digest in completion["checkpoint_hashes"].items():
        assert sha256(out / name) == digest
        check = completion["strict_reload"][name]
        assert check["strict_load"]
        if check["repeat_forward_max_error"] is not None:
            assert check["repeat_forward_max_error"] < 1e-7
    records = [json.loads(line) for line in (out / "evaluations.jsonl").open()]
    verified = []
    best, selected = float("inf"), None
    collapse_streak = 0
    stopped = False
    for record in records:
        tag = record["tag"]
        is_dev = tag.startswith("dev_")
        assert is_dev or tag.startswith("train_diagnostic_") or tag == "rhd_only_baseline"
        if is_dev:
            assert not stopped, "Dev evaluation after mandatory collapse stop"
        for domain in domains:
            with np.load(out / f"{tag}_{domain}_predictions.npz") as saved:
                if is_dev or tag == "rhd_only_baseline":
                    assert saved["sample_id"].tolist() == expected_ids[domain]
                m = record["domains"][domain]
                xyz, uv, cam, depth = [saved[k] for k in ("xyz", "uv", "cam", "depth")]
                finite = (np.isfinite(xyz).all((1, 2)) & np.isfinite(uv).all((1, 2))
                          & np.isfinite(cam).all(1) & np.isfinite(depth).all(1))
                span = np.max(np.max(uv, axis=1) - np.min(uv, axis=1), axis=1) * 256
                collapse = (span < config["training"]["collapse_threshold_px"]) | ~np.isfinite(span)
                negative = np.any(depth <= 0, axis=1)
                assert m["count"] == len(xyz)
                for name, value in {
                    "nonfinite_rate": np.mean(~finite),
                    "negative_depth_rate": np.mean(negative),
                    "projection_collapse_rate": np.mean(collapse),
                    "failure_rate": np.mean(~finite | negative | collapse),
                }.items():
                    close(m[name], value, f"{tag}/{domain}/{name}")
                if finite.any():
                    delta = xyz[finite].astype(np.float64) - saved["target_xyz"][finite]
                    error = np.sqrt(np.sum(delta * delta, axis=-1)) * 1000
                    close(m["wrist_relative_mpjpe_mm"], error.mean(), tag)
                    close(m["fingertip_wrist_relative_error_mm"],
                          error[:, [4, 8, 12, 16, 20]].mean(), tag)
                    for finger, name in enumerate(("thumb", "index", "middle", "ring", "little")):
                        close(m["per_finger_error_mm"][name],
                              error[:, 1+4*finger:5+4*finger].mean(), tag)
                    delta2 = uv[finite].astype(np.float64) - saved["target_uv"][finite]
                    epe = np.sqrt(np.sum(delta2 * delta2, axis=-1)) * 256
                    mask = saved["valid"][finite]
                    if mask.any():
                        close(m["visible_2d_epe_px"], epe[mask].mean(), tag)
                verified.append({"tag": tag, "domain": domain, "samples": len(xyz)})
        if is_dev:
            step = int(tag.rsplit("_", 1)[-1])
            collapsed = any(m["projection_collapse_rate"] > 0 for m in record["domains"].values())
            collapse_streak = collapse_streak + 1 if collapsed else 0
            stopped = collapse_streak >= config["training"]["collapse_stop_consecutive_evaluations"]
            if step:
                safe = all(m["failure_rate"] == 0 for m in record["domains"].values())
                value = record["domains"]["dexycb"].get("wrist_relative_mpjpe_mm", float("inf"))
                if safe and value < best - config["training"]["min_improvement_mm"]:
                    best, selected = value, step
    assert selected == completion["best_step"]
    assert completion == report["completion"]
    if stopped:
        assert completion["stop_reason"] == "consecutive_projection_collapse"
        assert not report["m_s2_eligible"]
    assert not completion["forbidden_actions_executed"]
    assert not completion["m_s2_started"]
    assert report["historical_smoke_failure"]["projection_collapse_rate"] == .09375
    result = {
        "passed": True, "model_calls": 0, "final_e4_data_reads": 0,
        "checks": ["copied code/config hashes", "split row and file hashes",
                   "Dex subject/sequence disjointness", "ordered full dev IDs",
                   "saved geometry/2D/failure metric recomputation",
                   "safe best checkpoint selection", "collapse stop",
                   "checkpoint hashes and recorded reload checks", "early failure retention"],
        "verified_evaluations": verified, "selected_step": selected,
        "limitations": ["Verifies saved evidence, not an independent proof of runtime access history.",
                        "Does not re-read source data or rerun inference."],
    }
    (out / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": True, "selected_step": selected, "model_calls": 0}), flush=True)


if __name__ == "__main__":
    main()
