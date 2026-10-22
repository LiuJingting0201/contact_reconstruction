"""Extract every RGB/label view from one official subject; freeze GT-valid rows."""
import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import numpy as np
from prepare_dexycb_subject_subset import (
    archive_audit, extract_calibration, safe_member, sha256,
)
import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=["train", "dev"], required=True)
    p.add_argument("--reuse-extracted", action="store_true",
                   help="Resume validation only after a completed extraction.")
    p.add_argument("--dev-sync-host", default="simready-dev",
                   help="For dev, stream the completed directory to this SSH host; empty disables.")
    a = p.parse_args()
    if not (a.audit / "archive_audit.json").exists():
        print(json.dumps({"stage": "archive_crc_audit", "archive": str(a.archive)}), flush=True)
        archive_audit(a.archive, a.audit)
    report = json.loads((a.audit / "archive_audit.json").read_text())
    expected = {"train": "20200813-subject-02", "dev": "20200928-subject-07"}
    assert report["subject"] == expected[a.split]
    assert sha256(a.archive) == report["source_sha256"]
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output / "sealed_manifest.json").exists():
        raise FileExistsError("Refusing to overwrite a frozen split")
    extracted = {}
    source = [] if a.reuse_extracted else tarfile.open(a.archive, "r|gz")
    print(json.dumps({"stage": "extracting", "reuse": a.reuse_extracted}), flush=True)
    try:
        for member in source:
            if not safe_member(member):
                raise ValueError(member.name)
            path = Path(member.name)
            if not member.isfile() or not (
                path.name == "meta.yml" or
                (len(path.parts) == 4 and
                 (path.name.startswith("color_") and path.suffix == ".jpg" or
                  path.name.startswith("labels_") and path.suffix == ".npz"))
            ):
                continue
            target = a.output / path
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as reader, target.open("wb") as writer:
                shutil.copyfileobj(reader, writer)
            extracted[member.name] = {"bytes": target.stat().st_size, "sha256": sha256(target)}
            if len(extracted) % 10000 == 0:
                print(json.dumps({"stage": "extracting", "files": len(extracted)}), flush=True)
    finally:
        if hasattr(source, "close"):
            source.close()
    if a.reuse_extracted:
        for target in a.output.rglob("*"):
            if target.is_file():
                extracted[str(target.relative_to(a.output))] = {
                    "bytes": target.stat().st_size, "sha256": sha256(target)}
    calibration = extract_calibration(a.calibration, a.output, extracted)
    expected = {
        f"{seq['sequence']}/{serial}/{kind}_{frame:06d}.{extension}"
        for seq in report["sequences"] for frame in range(seq["num_frames"])
        for serial in seq["serials"]
        for kind, extension in (("color", "jpg"), ("labels", "npz"))
    }
    missing = sorted(expected - extracted.keys())
    if missing:
        raise FileNotFoundError(f"Incomplete extraction: {len(missing)} missing; {missing[:10]}")
    Ks = {}
    samples, rejected, errors = [], [], []
    for seq in sorted(report["sequences"], key=lambda x: x["sequence"]):
        print(json.dumps({"stage": "validating", "sequence": seq["sequence"],
                          "accepted_times": len(samples), "rejected_views": len(rejected)}), flush=True)
        for frame in range(seq["num_frames"]):
            row = {
                "sample_id": f"{seq['sequence']}:{frame:06d}",
                "subject": report["subject"], "sequence": seq["sequence"],
                "frame_index": frame, "mano_side": seq["mano_sides"][0], "views": [],
            }
            for serial in seq["serials"]:
                image = f"{seq['sequence']}/{serial}/color_{frame:06d}.jpg"
                label = f"{seq['sequence']}/{serial}/labels_{frame:06d}.npz"
                if serial not in Ks:
                    intr = yaml.load((a.output / "calibration/intrinsics" /
                                      f"{serial}_640x480.yml").read_text(),
                                     Loader=yaml.FullLoader)["color"]
                    Ks[serial] = np.array([[intr["fx"], 0, intr["ppx"]],
                                          [0, intr["fy"], intr["ppy"]], [0, 0, 1]])
                reasons = []
                if image not in extracted or label not in extracted:
                    raise FileNotFoundError(f"Required view missing: {image}, {label}")
                else:
                    with np.load(a.output / label) as raw:
                        xyz, uv = raw["joint_3d"][0], raw["joint_2d"][0]
                        seg = raw["seg"]
                    valid = np.isfinite(uv).all(1) & (uv >= 0).all(1)
                    if not np.isfinite(xyz).all() or not (xyz[:, 2] > 0).all():
                        reasons.append("nonfinite_or_nonpositive_3d")
                    elif valid.sum() < 4:
                        reasons.append("fewer_than_four_2d_joints")
                    else:
                        h = xyz @ Ks[serial].T
                        error = np.linalg.norm(h[:, :2] / h[:, 2:] - uv, axis=1)[valid]
                        if error.max() >= 2:
                            reasons.append("projection_error_ge_2px")
                        errors.extend(error.tolist())
                if reasons:
                    rejected.append({"sample_id": row["sample_id"], "camera": serial,
                                     "reasons": reasons})
                    continue
                view = {"camera_serial": serial, "image": image, "label": label,
                        "valid_2d_joints": int(valid.sum()),
                        "occlusion_bucket": "partial" if valid.sum() < 21 else "all_annotated",
                        "hand_mask_fraction": float((seg == 255).mean()),
                        "image_sha256": extracted[image]["sha256"],
                        "label_sha256": extracted[label]["sha256"]}
                row["views"].append(view)
            if row["views"]:
                row["sha256"] = hashlib.sha256(
                    json.dumps(row, sort_keys=True).encode()).hexdigest()
                samples.append(row)
    manifest = {
        "schema": "dexycb_all_available_views_v1", "split": a.split,
        "subject": report["subject"], "source_archive": report,
        "selection_rule": "All frames, all official RGB cameras, all 100 sequences; "
                          "only fixed annotation-validity exclusion; no model access.",
        "sequence_count": len(report["sequences"]), "samples": samples,
        "views": sum(len(s["views"]) for s in samples),
        "synchronized_times": len(samples), "rejected_views": rejected,
        "calibration_archive": calibration, "extracted_files": extracted,
        "projection_error_px_mean": float(np.mean(errors)),
        "sealed_before_model_access": True,
    }
    path = a.output / "sealed_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    (a.output / "sealed_manifest.sha256").write_text(sha256(path) + "  sealed_manifest.json\n")
    print(json.dumps({"split": a.split, "times": len(samples), "views": manifest["views"],
                      "rejected": len(rejected), "sha256": sha256(path)}), flush=True)
    # The downstream orchestration rsync then only verifies metadata and fills
    # any differences. The currently active subject02 rsync is not interrupted.
    if a.split == "dev" and a.dev_sync_host:
        assert a.output.name == "20200928-subject-07"
        destination = "/workspace/pose_reconstract/inputs/dexycb_m_s1"
        print(json.dumps({"stage": "streaming_completed_dev", "host": a.dev_sync_host}), flush=True)
        producer = subprocess.Popen(
            ["tar", "-cf", "-", "-C", str(a.output.parent), a.output.name],
            stdout=subprocess.PIPE)
        consumer = subprocess.run(
            ["ssh", a.dev_sync_host,
             f"mkdir -p {destination} && tar -xf - -C {destination}"],
            stdin=producer.stdout)
        producer.stdout.close()
        code = producer.wait()
        if code or consumer.returncode:
            raise RuntimeError(f"dev archive stream failed: local={code}, remote={consumer.returncode}")
        print(json.dumps({"stage": "completed_dev_stream"}), flush=True)


if __name__ == "__main__":
    main()
