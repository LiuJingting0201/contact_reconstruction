"""Audit and selectively extract an official DexYCB subject archive.

Selection uses only sequence metadata and uniformly spaced frame indices. It
never reads images, labels, predictions, or model outputs while choosing rows.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile

import numpy as np
import yaml


CAMERAS = ("836212060125", "932122062010")
OFFICIAL_IDS = {
    "20200709-subject-01": "1Ehh92wDE3CWAiKG7E9E73HjN2Xk2XfEk",
    "20200813-subject-02": "1Uo7MLqTbXEa-8s7YQZ3duugJ1nXFEo62",
    "20200820-subject-03": "1FkUxas8sv8UcVGgAzmSZlJw1eI5W5CXq",
    "20200903-subject-04": "14up6qsTpvgEyqOQ5hir-QbjMB_dHfdpA",
    "20200908-subject-05": "1NBA_FPyGWOQF5-X9ueAat5g8lDMz-EmS",
    "20200918-subject-06": "1UWIN2-wOBZX2T0dkAi4ctAAW8KffkXMQ",
    "20200928-subject-07": "1oWEYD_o3PVh39pLzMlJcArkDtMj4nzI0",
    "20201002-subject-08": "1GTNZwhWbs7Mfez0krTgXwLPndvrw1Ztv",
    "20201015-subject-09": "1j0BLkaCjIuwjakmywKdOO9vynHTWR0UH",
    "20201022-subject-10": "1FvFlRfX-p5a5sAWoKEGc17zKJWwKaSB-",
}
CALIBRATION_ID = "1UAwVKT4Rgb1fLcFoa1o71_-0NtSvvLAQ"


class HashingReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.bytes = 0

    def read(self, size=-1):
        block = self.stream.read(size)
        self.digest.update(block)
        self.bytes += len(block)
        return block

    def readable(self):
        return True


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_member(member):
    path = Path(member.name)
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and not member.issym()
        and not member.islnk()
    )


def archive_audit(archive, audit_dir):
    if audit_dir.exists():
        raise FileExistsError(audit_dir)
    audit_dir.mkdir(parents=True)
    metadata = []
    members = 0
    with archive.open("rb") as raw_stream:
        hashing_stream = HashingReader(raw_stream)
        with gzip.GzipFile(fileobj=hashing_stream) as decompressed:
            with tarfile.open(fileobj=decompressed, mode="r|") as source:
                for member in source:
                    members += 1
                    if not safe_member(member):
                        raise ValueError(f"unsafe archive member: {member.name}")
                    parts = Path(member.name).parts
                    if (len(parts) == 3 and parts[-1] == "meta.yml"
                            and member.isfile()):
                        raw = source.extractfile(member).read()
                        meta = yaml.safe_load(raw)
                        metadata.append({
                            "sequence": "/".join(parts[:2]),
                            "num_frames": int(meta["num_frames"]),
                            "serials": list(meta["serials"]),
                            "mano_sides": list(meta["mano_sides"]),
                            "ycb_ids": [int(value) for value in meta["ycb_ids"]],
                            "ycb_grasp_ind": int(meta["ycb_grasp_ind"]),
                            "extrinsics": str(meta["extrinsics"]),
                            "mano_calib": list(meta["mano_calib"]),
                            "raw_sha256": hashlib.sha256(raw).hexdigest(),
                            "raw_text": raw.decode("utf-8"),
                        })
            # Force gzip trailer consumption and CRC validation after tar EOF.
            for _ in iter(lambda: decompressed.read(8 * 1024 * 1024), b""):
                pass
        archive_hash = hashing_stream.digest.hexdigest()
        if hashing_stream.bytes != archive.stat().st_size:
            raise ValueError("audit did not consume the complete compressed archive")
    if len(metadata) != 100:
        raise ValueError(f"expected 100 sequences, found {len(metadata)}")
    subjects = {row["sequence"].split("/")[0] for row in metadata}
    if len(subjects) != 1:
        raise ValueError(f"expected one subject, found {subjects}")
    report = {
        "source_archive": str(archive.resolve()),
        "source_bytes": archive.stat().st_size,
        "source_sha256": archive_hash,
        "official_url": (
            "https://drive.google.com/file/d/"
            + OFFICIAL_IDS.get(next(iter(subjects)), "UNKNOWN")
        ),
        "license": "CC BY-NC 4.0",
        "gzip_crc_eof_passed": True,
        "safe_members_scanned": members,
        "subject": next(iter(subjects)),
        "sequence_count": len(metadata),
        "selection_inputs": [
            "sequence name", "num_frames", "camera serials", "mano side",
            "YCB object IDs", "grasp index", "calibration IDs"
        ],
        "forbidden_selection_inputs": [
            "RGB pixels", "joint labels", "model output", "prediction error"
        ],
        "sequences": [{key: value for key, value in row.items() if key != "raw_text"}
                      for row in metadata],
    }
    (audit_dir / "archive_audit.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    meta_dir = audit_dir / "metadata"
    meta_dir.mkdir()
    for row in metadata:
        sequence = row["sequence"].split("/")[-1]
        (meta_dir / f"{sequence}.yml").write_text(row["raw_text"])
    return report


def choose_sequences(report, sequence_count):
    rows = sorted(report["sequences"], key=lambda row: row["sequence"])
    eligible = [
        row for row in rows
        if set(CAMERAS).issubset(row["serials"]) and row["num_frames"] > 0
    ]
    if len(eligible) < sequence_count:
        raise ValueError(f"only {len(eligible)} eligible sequences")
    # Even spacing across the official chronological sequence list gives broad
    # object/grasp coverage without looking at image or annotation quality.
    indices = [
        round(index * (len(eligible) - 1) / (sequence_count - 1))
        for index in range(sequence_count)
    ] if sequence_count > 1 else [len(eligible) // 2]
    chosen = [eligible[index] for index in indices]
    if len({row["sequence"] for row in chosen}) != sequence_count:
        raise ValueError("sequence spacing produced duplicates")
    return chosen


def choose_frames(num_frames, frames_per_sequence):
    if num_frames < frames_per_sequence:
        raise ValueError(f"{num_frames} frames cannot supply {frames_per_sequence}")
    if frames_per_sequence == 1:
        return [num_frames // 2]
    frames = [
        round(index * (num_frames - 1) / (frames_per_sequence - 1))
        for index in range(frames_per_sequence)
    ]
    if len(set(frames)) != frames_per_sequence:
        raise ValueError("frame spacing produced duplicates")
    return frames


def load_camera_calibration(output, sequence):
    intrinsics = {}
    for serial in CAMERAS:
        path = output / "calibration" / "intrinsics" / f"{serial}_640x480.yml"
        intr = yaml.load(path.read_text(), Loader=yaml.FullLoader)["color"]
        intrinsics[serial] = np.asarray([
            [intr["fx"], 0.0, intr["ppx"]],
            [0.0, intr["fy"], intr["ppy"]],
            [0.0, 0.0, 1.0],
        ])
    path = (
        output / "calibration" / f"extrinsics_{sequence['extrinsics']}"
        / "extrinsics.yml"
    )
    raw = yaml.load(path.read_text(), Loader=yaml.FullLoader)["extrinsics"]
    extrinsics = {}
    for serial in CAMERAS:
        transform = np.eye(4)
        transform[:3] = np.asarray(raw[serial]).reshape(3, 4)
        extrinsics[serial] = transform
    return intrinsics, extrinsics


def validate_candidates(output, candidates, sequence_lookup, frames_per_sequence):
    checks = []
    valid_by_sequence = {}
    for row in candidates:
        intrinsics, extrinsics = load_camera_calibration(
            output, sequence_lookup[row["sequence"]]
        )
        xyz_by_camera = {}
        projection_errors = []
        valid_joint_counts = []
        reasons = []
        for view in row["views"]:
            labels = np.load(output / view["label"])
            xyz = labels["joint_3d"]
            uv = labels["joint_2d"]
            if xyz.shape != (1, 21, 3) or uv.shape != (1, 21, 2):
                reasons.append(f"{view['camera_serial']}:unexpected_shape")
                continue
            xyz, uv = xyz[0].astype(float), uv[0].astype(float)
            if not np.isfinite(xyz).all() or not np.isfinite(uv).all():
                reasons.append(f"{view['camera_serial']}:nonfinite")
                continue
            if not (xyz[:, 2] > 0).all():
                reasons.append(f"{view['camera_serial']}:nonpositive_depth")
                continue
            valid = (uv >= 0).all(axis=1)
            valid_joint_counts.append(int(valid.sum()))
            if valid.sum() < 4:
                reasons.append(f"{view['camera_serial']}:fewer_than_4_valid_joints")
                continue
            projected_h = xyz @ intrinsics[view["camera_serial"]].T
            projected = projected_h[:, :2] / projected_h[:, 2:]
            error = np.linalg.norm(projected - uv, axis=1)[valid]
            projection_errors.extend(error.tolist())
            if error.max() >= 2.0:
                reasons.append(f"{view['camera_serial']}:projection_error_ge_2px")
            xyz_by_camera[view["camera_serial"]] = xyz
        stereo_error_mm = None
        if len(xyz_by_camera) == 2:
            left, right = CAMERAS
            transform = np.linalg.inv(extrinsics[right]) @ extrinsics[left]
            left_h = np.c_[xyz_by_camera[left], np.ones(21)]
            left_in_right = (left_h @ transform.T)[:, :3]
            stereo = np.linalg.norm(left_in_right - xyz_by_camera[right], axis=1)
            stereo_error_mm = float(stereo.mean() * 1000.0)
            if stereo.max() >= 0.002:
                reasons.append("stereo_3d_error_ge_2mm")
        passed = not reasons
        check = {
            "sample_id": row["sample_id"],
            "passed": passed,
            "reasons": reasons,
            "valid_joint_counts": valid_joint_counts,
            "projection_error_px_mean": (
                float(np.mean(projection_errors)) if projection_errors else None
            ),
            "projection_error_px_max": (
                float(np.max(projection_errors)) if projection_errors else None
            ),
            "stereo_3d_error_mm_mean": stereo_error_mm,
        }
        checks.append(check)
        if passed:
            valid_by_sequence.setdefault(row["sequence"], []).append(row)
    selected = []
    for sequence in sorted(sequence_lookup):
        valid = valid_by_sequence.get(sequence, [])
        if len(valid) < frames_per_sequence:
            raise ValueError(
                f"{sequence} has {len(valid)} valid candidates, "
                f"needs {frames_per_sequence}"
            )
        indices = choose_frames(len(valid), frames_per_sequence)
        selected.extend(valid[index] for index in indices)
    selected_ids = {row["sample_id"] for row in selected}
    for row in candidates:
        if row["sample_id"] in selected_ids:
            continue
        for view in row["views"]:
            for key in ("image", "label"):
                path = output / view[key]
                path.unlink()
    numeric_projection = [
        check["projection_error_px_mean"] for check in checks
        if check["projection_error_px_mean"] is not None
    ]
    numeric_stereo = [
        check["stereo_3d_error_mm_mean"] for check in checks
        if check["stereo_3d_error_mm_mean"] is not None
    ]
    audit = {
        "candidate_count": len(candidates),
        "passed_count": sum(check["passed"] for check in checks),
        "selected_count": len(selected),
        "selection_after_validation": (
            "uniform indices over chronological GT-valid candidates per sequence"
        ),
        "projection_error_px": {
            "mean": float(np.mean(numeric_projection)),
            "median": float(np.median(numeric_projection)),
            "p95": float(np.percentile(numeric_projection, 95)),
        },
        "stereo_3d_error_mm": {
            "mean": float(np.mean(numeric_stereo)),
            "median": float(np.median(numeric_stereo)),
            "p95": float(np.percentile(numeric_stereo, 95)),
        },
        "checks": checks,
    }
    return selected, audit


def extract_calibration(calibration, output, extracted):
    calibration_hash = sha256(calibration)
    with tarfile.open(calibration, "r:gz") as source:
        for member in source:
            if not safe_member(member):
                raise ValueError(f"unsafe calibration member: {member.name}")
            if not member.isfile():
                continue
            target = output / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as reader, target.open("wb") as writer:
                shutil.copyfileobj(reader, writer)
            extracted[member.name] = {
                "bytes": target.stat().st_size,
                "sha256": sha256(target),
            }
    return {
        "official_url": f"https://drive.google.com/file/d/{CALIBRATION_ID}",
        "bytes": calibration.stat().st_size,
        "sha256": calibration_hash,
    }


def extract_subset(archive, calibration, audit_dir, output, sequence_count,
                   frames_per_sequence, candidate_frames_per_sequence, split):
    if output.exists():
        raise FileExistsError(output)
    report = json.loads((audit_dir / "archive_audit.json").read_text())
    if sha256(archive) != report["source_sha256"]:
        raise ValueError("archive hash changed after audit")
    chosen = choose_sequences(report, sequence_count)
    desired = set()
    candidate_rows = []
    sequence_lookup = {row["sequence"]: row for row in chosen}
    for sequence in chosen:
        desired.add(f"{sequence['sequence']}/meta.yml")
        for frame in choose_frames(
                sequence["num_frames"], candidate_frames_per_sequence):
            views = []
            for serial in CAMERAS:
                image = f"{sequence['sequence']}/{serial}/color_{frame:06d}.jpg"
                label = f"{sequence['sequence']}/{serial}/labels_{frame:06d}.npz"
                desired.update((image, label))
                views.append({"camera_serial": serial, "image": image, "label": label})
            candidate_rows.append({
                "sample_id": f"{sequence['sequence']}:{frame:06d}",
                "subject": report["subject"],
                "sequence": sequence["sequence"],
                "frame_index": frame,
                "mano_side": sequence["mano_sides"][0],
                "ycb_ids": sequence["ycb_ids"],
                "ycb_grasp_ind": sequence["ycb_grasp_ind"],
                "views": views,
            })
    output.mkdir(parents=True)
    extracted = {}
    with tarfile.open(archive, "r|gz") as source:
        for member in source:
            if not safe_member(member):
                raise ValueError(f"unsafe archive member: {member.name}")
            if member.name not in desired:
                continue
            target = output / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if member.isfile():
                with source.extractfile(member) as reader, target.open("wb") as writer:
                    shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
                extracted[member.name] = {
                    "bytes": target.stat().st_size,
                    "sha256": sha256(target),
                }
    missing = desired - set(extracted)
    if missing:
        raise ValueError(f"archive entries missing: {sorted(missing)[:10]}")
    calibration_source = extract_calibration(calibration, output, extracted)
    sample_rows, gt_validation = validate_candidates(
        output, candidate_rows, sequence_lookup, frames_per_sequence
    )
    for name in list(extracted):
        if not (output / name).exists():
            del extracted[name]
    manifest = {
        "schema": "dexycb_subject_subset_v1",
        "split": split,
        "sealed_before_model_access": split == "final_e4",
        "source_archive": {
            "path": str(archive.resolve()),
            "official_url": report["official_url"],
            "bytes": report["source_bytes"],
            "sha256": report["source_sha256"],
            "gzip_crc_eof_passed": report["gzip_crc_eof_passed"],
        },
        "license": report["license"],
        "calibration_archive": calibration_source,
        "subject": report["subject"],
        "camera_serials": list(CAMERAS),
        "selection_rule": (
            "uniform sequence indices over all 100 official sequences; uniform frame "
            "indices including endpoints; metadata-only selection; no RGB, label, "
            "prediction, model, or error inspected"
        ),
        "sequence_count": sequence_count,
        "frames_per_sequence": frames_per_sequence,
        "candidate_frames_per_sequence": candidate_frames_per_sequence,
        "synchronized_times": len(sample_rows),
        "views": len(sample_rows) * len(CAMERAS),
        "gt_validation": gt_validation,
        "samples": sample_rows,
        "extracted_files": extracted,
    }
    manifest_path = output / "sealed_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    manifest_hash = sha256(manifest_path)
    (output / "sealed_manifest.sha256").write_text(
        f"{manifest_hash}  sealed_manifest.json\n"
    )
    if split == "final_e4":
        for path in output.rglob("*"):
            if path.is_file():
                os.chmod(path, 0o444)
        for path in sorted(
            (path for path in output.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            os.chmod(path, 0o555)
        os.chmod(output, 0o555)
    print(json.dumps({
        "output": str(output),
        "subject": report["subject"],
        "synchronized_times": len(sample_rows),
        "files": len(extracted),
        "manifest_sha256": manifest_hash,
    }))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--archive", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    extract = subparsers.add_parser("extract")
    extract.add_argument("--archive", type=Path, required=True)
    extract.add_argument("--calibration", type=Path, required=True)
    extract.add_argument("--audit", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--split", choices=("train", "dev", "final_e4"), required=True)
    extract.add_argument("--sequences", type=int, required=True)
    extract.add_argument("--frames-per-sequence", type=int, required=True)
    extract.add_argument("--candidate-frames-per-sequence", type=int)
    args = parser.parse_args()
    if args.command == "audit":
        report = archive_audit(args.archive, args.output)
        print(json.dumps({
            "subject": report["subject"],
            "source_sha256": report["source_sha256"],
            "sequences": report["sequence_count"],
        }))
    else:
        extract_subset(args.archive, args.calibration, args.audit, args.output,
                       args.sequences, args.frames_per_sequence,
                       args.candidate_frames_per_sequence or args.frames_per_sequence,
                       args.split)


if __name__ == "__main__":
    main()
