"""Export cached ViT+B2 without inference, optimization, or GT selection."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
JOINTS = ["wrist"] + [
    f"{finger}_{joint}" for finger in ("thumb", "index", "middle", "ring", "little")
    for joint in ("mcp", "pip", "dip", "tip")
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    source = ROOT / "artifacts/dex_pose_preserving_20261020_01"
    teacher = ROOT / "artifacts/dex_real_e2_matrix_20261020_02"
    metrics = json.loads((source / "independent_metrics_final.json").read_text())
    config = json.loads((source / "config.json").read_text())
    index = {(r["sample_id"], r["variant"]): r["metrics"] for r in metrics["rows"]}
    before = {}
    records = []
    states = []
    calibrations = []
    for folder in sorted(source.glob("sample_*")):
        row = json.loads((folder / "sample.json").read_text())
        state = np.load(folder / "B2/state.npz", allow_pickle=False)
        raw = np.load(folder / "B0/state.npz", allow_pickle=False)
        target = np.load(folder / "evaluation.npz", allow_pickle=False)
        for key in ("rotations", "betas"):
            assert np.array_equal(state[key], raw[key])
            assert np.array_equal(state[key], target[key])
        for key in ("K", "T"):
            assert np.array_equal(state[key], raw[key])
            assert np.array_equal(state[key], target[key].astype(state[key].dtype))
        xyz = state["xyz"]
        local = xyz - xyz[:1]
        K, T = target["K"], target["T"]
        camera = np.einsum("vab,jb->vja", T[:, :3, :3], xyz) + T[:, None, :3, 3]
        h = np.einsum("vab,vjb->vja", K, camera)
        uv = h[..., :2] / h[..., 2:]
        np.testing.assert_allclose(uv, state["uv"], atol=.003, rtol=1e-5)
        finite = bool(np.isfinite(xyz).all() and np.isfinite(uv).all())
        positive = bool((camera[..., 2] > config["minimum_depth_m"]).all())
        assert finite and positive
        runtime = json.loads((folder / "B2/runtime.json").read_text())
        assert runtime["numerical_failure"] is None
        assert runtime["optimizer_steps"] == 800
        # These are evaluation-only annotations; they never select or filter records.
        audit = index[row["sample_id"], "B2"]
        cameras = []
        for v, metadata in enumerate(row["views"]):
            assert np.array_equal(K[v], np.asarray(metadata["K"]))
            np.testing.assert_allclose(T[v], np.asarray(metadata["T_world_to_camera"]),
                                       atol=1e-12, rtol=1e-12)
            cameras.append({
                "role": metadata["role"], "camera_serial": metadata["camera_serial"],
                "resolution_wh": metadata["resolution_wh"], "K": K[v].tolist(),
                "T_left_to_camera": T[v].tolist(),
                "distortion": metadata["distortion"],
                "image_sha256": metadata["image_sha256"],
                "crop_box_xyxy": metadata["crop_box_xyxy"],
                "extrinsics_source_sha256": metadata["extrinsics_source_sha256"],
            })
        record = {
            "sample_id": row["sample_id"], "sequence_id": row["sequence_id"],
            "source_original_id": row["source_original_id"],
            "frame_index": row["frame_index"], "sync_id": row["sync_id"],
            "timestamp_s": row["timestamp_s"],
            "timestamp_kind": "nominal_sequence_relative_frame_index_divided_by_30",
            "handedness": row["handedness"],
            "reference_frame": row["views"][0]["reference_frame"],
            "joints_left_camera_m": xyz.tolist(),
            "joints_wrist_local_m": local.tolist(),
            "global_orientation_matrix": state["rotations"][0, 0].tolist(),
            "articulated_rotation_matrices": state["rotations"][0, 1:].tolist(),
            "shape_betas": state["betas"][0].tolist(),
            "mano_translation_left_camera_m": state["translation"][0].tolist(),
            "confidence": None,
            "confidence_source": "unavailable: HaMeR does not output learned confidence",
            "quality_flag": "finite_positive_depth" if finite and positive else "invalid",
            "quality_flag_uses_gt": False,
            "evaluation_only": {"quality_failure_gt_assisted": bool(audit["quality_failure"])},
            "cameras": cameras,
        }
        records.append(record)
        states.append({key: state[key].copy() for key in state.files})
        calibrations.append((K, T))
        # Preserve exact raw predictions and compact optimization states/trajectories.
        dst = out / "cached_evidence" / folder.name
        dst.mkdir(parents=True)
        paths = [folder / "sample.json", folder / "evaluation.npz",
                 teacher / folder.name / "teacher.npz"]
        for variant in ("B0", "B2"):
            paths += [folder / variant / name for name in ("state.npz", "runtime.json")]
        for path in paths:
            before[str(path.relative_to(ROOT))] = sha(path)
            if path.parent.name in ("B0", "B2"):
                dest = dst / path.parent.name / path.name
                dest.parent.mkdir(exist_ok=True)
            else:
                dest = dst / path.name
            shutil.copy2(path, dest)
            assert sha(path) == sha(dest)
    ids = [r["sample_id"] for r in records]
    assert len(ids) == len(set(ids)) == 32
    assert ids == [json.loads((f / "sample.json").read_text())["sample_id"]
                   for f in sorted(teacher.glob("sample_*"))]
    write(out / "retargeting.json", {"schema_version": "1.0", "joint_names": JOINTS,
                                     "pipeline": "HaMeR ViT + frozen B2", "records": records})
    np.savez_compressed(
        out / "retargeting.npz",
        sample_id=np.asarray(ids),
        sequence_id=np.asarray([r["sequence_id"] for r in records]),
        frame_index=np.asarray([r["frame_index"] for r in records]),
        timestamp_s=np.asarray([r["timestamp_s"] for r in records]),
        handedness=np.asarray([r["handedness"] for r in records]),
        joint_names=np.asarray(JOINTS),
        joints_left_camera_m=np.stack([s["xyz"] for s in states]),
        joints_wrist_local_m=np.asarray([r["joints_wrist_local_m"] for r in records],
                                       dtype=np.float32),
        global_orientation_matrix=np.stack([s["rotations"][0, 0] for s in states]),
        articulated_rotation_matrices=np.stack([s["rotations"][0, 1:] for s in states]),
        shape_betas=np.stack([s["betas"][0] for s in states]),
        mano_translation_left_camera_m=np.stack([s["translation"][0] for s in states]),
        K=np.stack([c[0] for c in calibrations]),
        T_left_to_camera=np.stack([c[1] for c in calibrations]),
        camera_serial=np.asarray([[c["camera_serial"] for c in r["cameras"]] for r in records]),
        confidence_available=np.zeros(32, dtype=bool),
        quality_flag=np.asarray([r["quality_flag"] for r in records]),
        evaluation_only_quality_failure_gt_assisted=np.asarray([
            r["evaluation_only"]["quality_failure_gt_assisted"] for r in records]),
    )
    loaded = np.load(out / "retargeting.npz", allow_pickle=False)
    decoded = json.loads((out / "retargeting.json").read_text())["records"]
    for i, record in enumerate(decoded):
        for key in ("joints_left_camera_m", "joints_wrist_local_m", "global_orientation_matrix"):
            assert np.array_equal(loaded[key][i], np.asarray(record[key], dtype=np.float32))
        assert np.array_equal(loaded["joints_left_camera_m"][i], states[i]["xyz"])
        assert np.array_equal(loaded["joints_wrist_local_m"][i],
                              states[i]["xyz"] - states[i]["xyz"][:1])
    for file in ("config.json", "source_provenance.json", "input_manifest.json"):
        shutil.copy2(source / file, out / file)
        before[str((source / file).relative_to(ROOT))] = sha(source / file)
    shutil.copy2(teacher / "teacher_provenance.json", out / "teacher_provenance.json")
    write(out / "baseline_metrics.json", {
        "status": "formal_E3_blocked_missing_author_student_checkpoint",
        "A": {"definition": "cached B0: shared left MANO + real-K root initialization",
              "aggregates": metrics["aggregates"]["B0"]},
        "B": {"definition": "cached B2, unchanged", "aggregates": metrics["aggregates"]["B2"]},
        "C": {"status": "blocked"}, "D": {"status": "blocked"},
        "rows": [r for r in metrics["rows"] if r["variant"] in ("B0", "B2")],
        "speed_compensation_memory_ratios": None,
        "timing_status": "not_benchmarked_under_E3_protocol",
    })
    for path, digest in before.items():
        assert sha(ROOT / path) == digest
    write(out / "source_hashes.json", before)
    write(out / "verification.json", {
        "passed": True, "sample_count": 32, "unique_ordered_ids_match": True,
        "frozen_rotation_shape_exact": True, "calibration_unchanged": True,
        "export_equals_cached_B2": True, "json_npz_roundtrip": True,
        "finite_positive_depth_samples": 32,
        "gt_assisted_quality_failures_retained": sum(
            r["evaluation_only"]["quality_failure_gt_assisted"] for r in records),
        "inference_calls": 0, "optimization_calls": 0, "source_files_unchanged": True,
        "checkpoint_reload_evidence": "prior independent verification; exact state copied",
    })
    write(out / "sha256.json", {
        str(f.relative_to(out)): sha(f) for f in sorted(out.rglob("*")) if f.is_file()
    })
    print(json.dumps({"output": str(out), "samples": len(records), "verified": True}))


if __name__ == "__main__":
    main()
