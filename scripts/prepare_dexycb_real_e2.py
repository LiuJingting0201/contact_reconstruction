"""Official DexYCB subset loader, RGB detector and camera-frame calibration audit.

Only dataset-selection lines of the installed official initializer are adapted:
present subjects/sequences instead of all ten complete subjects. All file formats,
metadata parsing, joint order, intrinsics and __getitem__ remain official.
"""
import argparse
import difflib
import inspect
import json
import os
from pathlib import Path
import textwrap
import numpy as np
import cv2
import yaml
from real_e2_data import JOINT_ORDER, sha256, require, transform, project

ROOT = Path(__file__).resolve().parents[1]
PIN = "64551b001d360ad83bc383157a559ec248fb9100"
BASE = f"https://github.com/NVlabs/dex-ycb-toolkit/blob/{PIN}"
GUIDE = "https://docs.google.com/document/d/1jflhWGTn4lsGj5PiJ14Q37k0xHSnkTuDJvkdFr_j-fo"


def official_subset(root, output):
    import dex_ycb_toolkit.dex_ycb as official
    os.environ["DEX_YCB_DIR"] = str(root)
    original = textwrap.dedent(inspect.getsource(official.DexYCBDataset.__init__))
    modified = original.replace(
        "self._subjects = [_SUBJECTS[i] for i in subject_ind]",
        "self._subjects = [_SUBJECTS[i] for i in subject_ind if os.path.isdir(os.path.join(self._data_dir, _SUBJECTS[i]))]"
    ).replace("assert len(seq) == 100", "assert len(seq) > 0").replace(
        "seq = [seq[i] for i in sequence_ind]",
        "seq = [seq[i] for i in range(len(seq))]"
    )
    require(modified != original, "official initializer changed; audit selection adapter")
    namespace = dict(vars(official))
    exec(compile(modified, str(output/"official_subset_initializer.py"), "exec"), namespace)
    class Subset(official.DexYCBDataset):
        __init__ = namespace["__init__"]
    (output/"official_initializer.py").write_text(original)
    (output/"official_subset_initializer.py").write_text(modified)
    (output/"official_selection_only.diff").write_text("".join(difflib.unified_diff(
        original.splitlines(True), modified.splitlines(True), "official", "subset_selection")))
    return Subset("s0", "train")


def detector_survey(dataset, out):
    import mediapipe as mp
    raw = {}
    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, model_complexity=1,
                                 min_detection_confidence=.3) as detector:
        for idx in range(len(dataset)):
            sample = dataset[idx]
            if sample["mano_side"] != "right":
                continue
            seq, cam, frame = map(int, dataset._mapping[idx])
            path = Path(sample["color_file"])
            image = cv2.imread(str(path))
            require(image is not None, f"missing official RGB {path}")
            prediction = detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            candidates = []
            if prediction.multi_hand_landmarks:
                for landmarks, handed in zip(prediction.multi_hand_landmarks, prediction.multi_handedness):
                    uv = np.array([[x.x*image.shape[1], x.y*image.shape[0]] for x in landmarks.landmark])
                    h = handed.classification[0]
                    # MediaPipe labels assume mirrored selfies. For these original
                    # RGB frames, swap the reported physical handedness.
                    side = "right" if h.label == "Left" else "left"
                    candidates.append({"uv": uv.tolist(), "side": side, "score": float(h.score)})
            rights = [x for x in candidates if x["side"] == "right"]
            selected = max(rights, key=lambda x:x["score"]) if rights else None
            raw[idx] = {"sequence": dataset._sequences[seq], "camera": dataset._serials[cam],
                        "frame": frame, "candidates": candidates, "selected": selected}
            if idx % 100 == 0:
                print(json.dumps({"detector_index": idx, "total": len(dataset)}), flush=True)
    package = Path(mp.__file__).parent
    hashes = {str(p.relative_to(package)): sha256(p) for p in package.rglob("*.tflite")
              if "hand" in str(p) or "palm" in str(p)}
    (out/"detector_predictions.json").write_text(json.dumps(raw, indent=2)+"\n")
    (out/"detector_provenance.json").write_text(json.dumps({
        "model": "MediaPipe Hands", "version": mp.__version__, "model_complexity": 1,
        "static_image_mode": True, "detection_threshold": .3, "model_hashes": hashes,
        "box_rule": "2.0x square around predicted landmarks; no evaluation GT",
        "handedness": "swap selfie labels on original unmirrored RGB"
    }, indent=2)+"\n")
    return raw


def make_manifest(dataset, predictions, out, count):
    mapping = [JOINT_ORDER.index(name) for name in dataset.mano_joints]
    require(mapping == list(range(21)), "nonidentity mapping needs explicit reorder implementation")
    (out/"joint_mapping.json").write_text(json.dumps({
        "official_joint_order": dataset.mano_joints, "project_joint_order": JOINT_ORDER,
        "project_from_official_indices": mapping,
        "source": BASE+"/dex_ycb_toolkit/dex_ycb.py#L58-L81"
    }, indent=2)+"\n")
    camera_pairs, per_sequence, invalid_gt = [], {}, []
    for seq_id, seq in enumerate(dataset._sequences):
        meta = yaml.safe_load((Path(dataset.data_dir)/seq/"meta.yml").read_text())
        if meta["mano_sides"][0] != "right":
            continue
        calibration_file = Path(dataset.data_dir)/"calibration"/f"extrinsics_{meta['extrinsics']}"/"extrinsics.yml"
        extr = yaml.load(calibration_file.read_text(), Loader=yaml.FullLoader)
        C = {}
        for serial in meta["serials"]:
            C[serial] = np.eye(4)
            C[serial][:3] = np.asarray(extr["extrinsics"][serial]).reshape(3, 4)
        by_cam = {}
        for idx, record in predictions.items():
            if record["sequence"] == seq and record["selected"] is not None:
                annotation = np.load(dataset[idx]["label_file"])
                xyz = annotation["joint_3d"]
                uv = annotation["joint_2d"]
                if not (np.isfinite(xyz).all() and (xyz[..., 2] > 0).all()
                        and ((uv[0] >= 0).all(-1).sum() >= 4)):
                    invalid_gt.append({"index": idx, "sequence": seq, "camera": record["camera"],
                                       "frame": record["frame"], "reason": "official missing/invalid annotation"})
                    continue
                by_cam.setdefault(record["camera"], {})[record["frame"]] = idx
        candidates = []
        for a, left in enumerate(meta["serials"]):
            for right in meta["serials"][a+1:]:
                frames = sorted(set(by_cam.get(left, {})) & set(by_cam.get(right, {})))
                baseline = np.linalg.norm(C[left][:3, 3]-C[right][:3, 3])
                # Rank by detector joint presence only, then baseline. No teacher
                # accuracy/optimization/GT residual participates in this selection.
                if len(frames) >= count and baseline >= .15:
                    candidates.append((len(frames), baseline, left, right, frames))
        require(candidates, f"no camera pair with {count} jointly detected right-hand frames in {seq}")
        _, baseline, left, right, frames = max(candidates)
        chosen = [frames[j] for j in np.linspace(0, len(frames)-1, count).astype(int)]
        camera_pairs.append({"sequence": seq, "left": left, "right": right, "baseline_m": float(baseline),
                             "all_jointly_detected_frames": frames, "selected_frames": chosen,
                             "selection": "maximum jointly detected frames, baseline tiebreak; evenly spaced frames"})
        rows = []
        for frame in chosen:
            views, original_gt = [], []
            left_C = C[left]
            for role, serial in (("left", left), ("right", right)):
                idx = by_cam[serial][frame]
                entry = dataset[idx]  # unmodified official __getitem__
                labels = np.load(entry["label_file"])
                require(labels["joint_3d"].shape == (1, 21, 3), "unexpected hand GT shape")
                xyz, uv = labels["joint_3d"][0].astype(float), labels["joint_2d"][0].astype(float)
                require(np.isfinite(xyz).all() and (xyz[:, 2] > 0).all(), "invalid official GT3D in selected sample")
                intr = entry["intrinsics"]
                K = np.array([[intr["fx"], 0, intr["ppx"]], [0, intr["fy"], intr["ppy"]], [0, 0, 1.]])
                predicted_uv = np.array(predictions[idx]["selected"]["uv"])
                extent = predicted_uv.max(0)-predicted_uv.min(0)
                side = float(max(extent)*2)
                center = (predicted_uv.max(0)+predicted_uv.min(0))/2
                box = np.r_[center-side/2, center+side/2]
                T = np.linalg.inv(C[serial]) @ left_C
                direct, _ = project(xyz, K, np.eye(4))
                valid = (uv >= 0).all(-1)
                require(valid.sum() >= 4, "insufficient official 2D labels")
                require(np.linalg.norm(direct-uv, axis=-1)[valid].max() < 2., "official pinhole GT mismatch")
                original_gt.append(xyz)
                raw_out = out/"raw_labels"/seq.split("/")[-1]/serial
                raw_out.mkdir(parents=True, exist_ok=True)
                # Preserve exact dataset label file, including pose_m and segmentation.
                import shutil
                shutil.copy2(entry["label_file"], raw_out/Path(entry["label_file"]).name)
                view = {
                    "role": role, "image": entry["color_file"], "image_sha256": sha256(entry["color_file"]),
                    "depth_file": entry["depth_file"], "label_file": entry["label_file"],
                    "depth_sha256": sha256(entry["depth_file"]), "label_sha256": sha256(entry["label_file"]),
                    "camera_serial": serial, "frame_index": frame, "timestamp_s": frame/30,
                    "sync_id": str(frame), "resolution_wh": [640, 480], "K_resolution_wh": [640, 480],
                    "K": K.tolist(), "T_world_to_camera": T.tolist(),
                    "reference_frame": f"left_camera:{left}", "gt3d": xyz.tolist(),
                    "pose_m": labels["pose_m"].tolist(), "mano_side": entry["mano_side"],
                    "gt2d": uv.tolist(), "gt2d_valid": valid.tolist(),
                    "gt2d_origin": "official_dataset_joint_2d",
                    "distortion": {"state": "dataset_delivered_pinhole", "model": "dataset_pinhole",
                                   "coefficients": None,
                                   "evidence": [GUIDE, BASE+"/dex_ycb_toolkit/sequence_loader.py#L69-L82"]},
                    "undistort_count": 0, "mirror_count": 0,
                    "crop_box_xyxy": box.tolist(), "crop_source": "predicted_detector",
                    "crop_coordinate_system": "undistorted_full_image",
                    "oracle_visibility": None,
                    "oracle_visibility_source": None,
                    "extrinsics_source": str(calibration_file),
                    "extrinsics_source_sha256": sha256(calibration_file)
                }
                views.append(view)
            # Independently check supplied direction AND inverse alternative.
            right_to_left = np.linalg.inv(left_C) @ C[right]
            aligned = transform(original_gt[1], right_to_left)
            cross_error = np.linalg.norm(aligned-original_gt[0], axis=-1)
            wrong = np.linalg.norm(transform(original_gt[1], np.linalg.inv(right_to_left))-original_gt[0], axis=-1)
            require(cross_error.max() < .005, "cross-camera independent GT disagrees >5mm")
            require(cross_error.mean() < wrong.mean(), "extrinsic direction ambiguous")
            rows.append({"sample_id": seq.split("/")[-1]+f"_{frame:06d}", "sequence_id": seq,
                         "source_original_id": seq+f"/frame_{frame:06d}",
                         "sync_id": str(frame), "frame_index": frame, "timestamp_s": frame/30,
                         "sync_evidence": GUIDE+"; official same-frame synchronized export",
                         "split": "method_validation", "handedness": "right", "gt3d_valid": [True]*21,
                         "coverage_tags": ["hand_object_interaction"],
                         "cross_camera_gt_max_m": float(cross_error.max()),
                         "wrong_inverse_gt_mean_m": float(wrong.mean()), "views": views})
        per_sequence[seq] = rows
    rows = [row for seq in sorted(per_sequence) for row in per_sequence[seq]]
    require(20 <= len(rows) <= 50, "need 20–50 samples")
    manifest = {
        "schema": "real_e2_v1", "kind": "real", "units": "m", "gt_coordinate_system": "per_camera",
        "joint_order": JOINT_ORDER, "timestamp_kind": "dataset_frame_index_over_fps",
        "sync_tolerance_s": None, "synchronization_code_evidence": GUIDE,
        "synchronization_limit": "guide uses ApproximateTimeSynchronizer slop=0.1; physical timestamps not exported",
        "source": {"name": "official DexYCB subject-01", "url": "https://dex-ycb.github.io/",
                   "toolkit_commit": PIN, "license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
                   "license_allows_experiment": True, "gt_independent_of_teacher": True,
                   "gt_method": "official multi-view RGBD/MANO hand annotations; independent of current HaMeR"},
        "visual_sync_review": {"passed": False, "evidence": "pending actual overlay inspection"},
        "samples": rows
    }
    (out/"camera_selection.json").write_text(json.dumps(camera_pairs, indent=2)+"\n")
    (out/"invalid_gt_exclusions.json").write_text(json.dumps(invalid_gt, indent=2)+"\n")
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--frames-per-sequence", type=int, default=16)
    p.add_argument("--detections", type=Path)
    a = p.parse_args()
    out = ROOT/"artifacts"/a.run_id
    out.mkdir(exist_ok=False)
    import shutil
    shutil.copy2(__file__, out/Path(__file__).name)
    dataset = official_subset(a.data.resolve(), out)
    if a.detections:
        predictions = {int(k): v for k, v in json.loads(a.detections.read_text()).items()}
        shutil.copy2(a.detections, out/"detector_predictions.json")
        shutil.copy2(a.detections.parent/"detector_provenance.json", out/"detector_provenance.json")
    else:
        predictions = detector_survey(dataset, out)
    manifest = make_manifest(dataset, predictions, out, a.frames_per_sequence)
    from real_e2_data import load_manifest
    config = json.loads((ROOT/"configs/real_e2_dexycb_official_v2.json").read_text())
    _, _, checks = load_manifest(out/"manifest.json", config, out/"gt_overlays")
    (out/"validation.json").write_text(json.dumps(checks, indent=2)+"\n")
    print(json.dumps({"prepared": len(manifest["samples"]), "manifest": str(out/"manifest.json")}), flush=True)


if __name__ == "__main__":
    main()
