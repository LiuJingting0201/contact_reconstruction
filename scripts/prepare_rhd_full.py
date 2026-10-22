"""Prepare all usable RHD hand instances and perform the R0 coordinate audit."""
import argparse
import hashlib
import json
import pickle
import struct
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "inputs/rhd"
ORDER = np.asarray([0, 4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9,
                    16, 15, 14, 13, 20, 19, 18, 17])
FINGERS = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
           (0, 13, 14, 15, 16), (0, 17, 18, 19, 20))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_cached(entry):
    relative = entry["name"].removeprefix("RHD_published_v2/")
    path = DATA / relative
    if path.is_file() and zlib.crc32(path.read_bytes()) == entry["crc"]:
        return True
    return False


def decode_entry(entry, blob, range_start):
    relative = entry["name"].removeprefix("RHD_published_v2/")
    offset = entry["offset"] - range_start
    fields = struct.unpack("<IHHHHHIIIHH", blob[offset:offset + 30])
    assert fields[0] == 0x04034B50 and fields[3] in (0, 8)
    payload_start = offset + 30 + fields[-2] + fields[-1]
    payload = blob[payload_start:payload_start + entry["compressed"]]
    value = zlib.decompress(payload, -15) if fields[3] == 8 else payload
    assert len(value) == entry["bytes"]
    assert zlib.crc32(value) == entry["crc"]
    path = DATA / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return len(value)


def fetch_batch(entries, url):
    missing = [entry for entry in entries if not valid_cached(entry)]
    if not missing:
        return {"cached": len(entries), "fetched": 0,
                "bytes": sum(entry["bytes"] for entry in entries)}
    start = entries[0]["offset"]
    last = entries[-1]
    end = last["offset"] + 30 + len(last["name"].encode()) + 4096 + last["compressed"] - 1
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    last_error = None
    for _ in range(4):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                assert response.status == 206
                blob = response.read()
            total = sum(decode_entry(entry, blob, start) for entry in missing)
            return {"cached": len(entries) - len(missing), "fetched": len(missing),
                    "bytes": total + sum(entry["bytes"] for entry in entries
                                         if entry not in missing)}
        except Exception as error:
            last_error = error
    raise RuntimeError(f"failed batch {entries[0]['name']}..{entries[-1]['name']}: {last_error}")


def make_batches(entries, maximum_bytes=32 * 1024 * 1024):
    batches, current = [], []
    for entry in sorted(entries, key=lambda item: item["offset"]):
        if current and entry["offset"] + entry["compressed"] - current[0]["offset"] > maximum_bytes:
            batches.append(current)
            current = []
        current.append(entry)
    if current:
        batches.append(current)
    return batches


def split_eval(image_id, seed):
    digest = hashlib.sha256(f"{seed}:{image_id}".encode()).digest()
    return "dev" if int.from_bytes(digest[:8], "big") % 10 < 7 else "test"


def project(xyz, K):
    homogeneous = xyz @ K.T
    return homogeneous[:, :2] / homogeneous[:, 2:]


def build_examples(source_split, annotations, min_visible, min_extent, seed):
    rows = []
    for image_id in sorted(annotations):
        annotation = annotations[image_id]
        K = np.asarray(annotation["K"], np.float64)
        for hand_index, hand in enumerate(("left", "right")):
            source = slice(hand_index * 21, (hand_index + 1) * 21)
            uv_vis = np.asarray(annotation["uv_vis"][source])[ORDER].astype(np.float32)
            xyz = np.asarray(annotation["xyz"][source])[ORDER].astype(np.float32)
            visible = uv_vis[:, 2] > 0
            if visible.sum() < min_visible or not (xyz[:, 2] > 0).all():
                continue
            visible_uv = uv_vis[visible, :2]
            extent = float(np.ptp(visible_uv, axis=0).max())
            if extent < min_extent:
                continue
            reprojection = np.linalg.norm(project(xyz.astype(np.float64), K) -
                                          uv_vis[:, :2], axis=1)
            output_split = "train" if source_split == "training" else split_eval(image_id, seed)
            rows.append({
                "sample_id": f"rhd-{source_split}-{image_id:05d}-{hand}",
                "source_split": source_split,
                "split": output_split,
                "image_id": int(image_id),
                "hand": hand,
                "right": hand == "right",
                "image": f"{source_split}/color/{image_id:05d}.png",
                "uv_vis": uv_vis,
                "xyz_m": xyz,
                "K": K.astype(np.float32),
                "source_projection_max_px": float(reprojection.max()),
            })
    return rows


def crop_record(row, image_size, rescale):
    visible = row["uv_vis"][:, 2] > 0
    uv = row["uv_vis"][:, :2]
    lo, hi = uv[visible].min(0), uv[visible].max(0)
    center = (lo + hi) / 2
    side = max(float((hi - lo).max() * rescale), 2.0)
    scale = image_size / side
    affine = np.asarray([[scale, 0, image_size / 2 - scale * center[0]],
                         [0, scale, image_size / 2 - scale * center[1]]], np.float32)
    crop_uv = np.c_[uv, np.ones(21)] @ affine.T
    canonical_xyz = row["xyz_m"] - row["xyz_m"][:1]
    if not row["right"]:
        crop_uv[:, 0] = image_size - 1 - crop_uv[:, 0]
        canonical_xyz[:, 0] *= -1
    return affine, crop_uv.astype(np.float32), canonical_xyz.astype(np.float32)


def save_npz(rows, path, image_size, rescale):
    affine, crop_uv, xyz = zip(*(crop_record(row, image_size, rescale) for row in rows))
    np.savez_compressed(
        path,
        sample_id=np.asarray([row["sample_id"] for row in rows]),
        image=np.asarray([row["image"] for row in rows]),
        image_id=np.asarray([row["image_id"] for row in rows]),
        right=np.asarray([row["right"] for row in rows]),
        uv_vis=np.stack([row["uv_vis"] for row in rows]),
        xyz_absolute_m=np.stack([row["xyz_m"] for row in rows]),
        xyz_canonical_wrist_relative_m=np.stack(xyz),
        K=np.stack([row["K"] for row in rows]),
        affine_full_to_crop=np.stack(affine),
        uv_canonical_crop_px=np.stack(crop_uv),
    )


def overlay_audit(rows, name, out, image_size, rescale, seed):
    rng = np.random.default_rng(seed + {"train": 0, "dev": 1, "test": 2}[name])
    selected = sorted(rng.choice(len(rows), 32, replace=False).tolist())
    tiles, audits = [], []
    for index in selected:
        row = rows[index]
        image = cv2.imread(str(DATA / row["image"]))
        assert image is not None and image.shape[:2] == (320, 320)
        affine, uv, canonical_xyz = crop_record(row, image_size, rescale)
        crop = cv2.warpAffine(image, affine, (image_size, image_size),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if not row["right"]:
            crop = crop[:, ::-1].copy()
        source_uv = row["uv_vis"][:, :2]
        check_uv = uv.copy()
        if not row["right"]:
            check_uv[:, 0] = image_size - 1 - check_uv[:, 0]
        recovered = np.c_[check_uv, np.ones(21)] @ cv2.invertAffineTransform(affine).T
        inverse_error = float(np.abs(recovered - source_uv).max())
        assert inverse_error < 1e-3
        assert np.isfinite(canonical_xyz).all()
        for finger in FINGERS:
            points = np.asarray([uv[j] for j in finger], np.int32)
            cv2.polylines(crop, [points], False, (0, 255, 255), 1, cv2.LINE_AA)
        for joint, ((x, y), visible) in enumerate(zip(uv, row["uv_vis"][:, 2] > 0)):
            color = (0, 255, 0) if visible else (0, 128, 255)
            cv2.circle(crop, (round(float(x)), round(float(y))), 2, color, -1)
            cv2.putText(crop, str(joint), (round(float(x)) + 2, round(float(y))),
                        cv2.FONT_HERSHEY_SIMPLEX, .23, color, 1, cv2.LINE_AA)
        cv2.putText(crop, f"{row['image_id']:05d} {row['hand']}", (4, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, .35, (255, 255, 255), 1)
        tiles.append(crop)
        bone_lengths = np.concatenate([
            np.linalg.norm(np.diff(canonical_xyz[list(finger)], axis=0), axis=1)
            for finger in FINGERS
        ])
        audits.append({
            "sample_id": row["sample_id"],
            "source_projection_max_px": row["source_projection_max_px"],
            "crop_inverse_max_px": inverse_error,
            "visible_joints": int((row["uv_vis"][:, 2] > 0).sum()),
            "canonical_bone_length_min_max_m": [float(bone_lengths.min()),
                                                 float(bone_lengths.max())],
        })
    contact = np.concatenate([
        np.concatenate(tiles[row * 8:(row + 1) * 8], axis=1) for row in range(4)
    ], axis=0)
    cv2.imwrite(str(out / f"{name}_r0_overlay_32.jpg"), contact)
    return audits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-visible", type=int, default=4)
    parser.add_argument("--min-extent-px", type=float, default=20)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--crop-rescale", type=float, default=2.0)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--archive-url", help="Override the archive URL, e.g. a signed mirror URL")
    args = parser.parse_args()
    out = ROOT / "artifacts/rhd_full_r0_20261020_01"
    out.mkdir(parents=True, exist_ok=True)
    index_path = ROOT / "artifacts/rhd_archive_index.json"
    archive = json.loads(index_path.read_text())
    archive_url = args.archive_url or archive["url"]
    entries = {entry["name"]: entry for entry in archive["entries"]}
    annotations = {}
    rows = []
    for split in ("training", "evaluation"):
        annotation_path = DATA / split / f"anno_{split}.pickle"
        annotations[split] = pickle.loads(annotation_path.read_bytes())
        rows.extend(build_examples(split, annotations[split], args.min_visible,
                                   args.min_extent_px, args.seed))
    required_images = sorted({row["image"] for row in rows})
    required_entries = [entries["RHD_published_v2/" + relative]
                        for relative in required_images]
    batches = make_batches(required_entries)
    counters = {"cached": 0, "fetched": 0, "bytes": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {
            pool.submit(fetch_batch, batch, archive_url): batch for batch in batches
        }
        for number, job in enumerate(as_completed(jobs), 1):
            result = job.result()
            for key in counters:
                counters[key] += result[key]
            print(json.dumps({"completed_batches": number, "total_batches": len(jobs),
                              "total_images": len(required_images), **counters}), flush=True)
    if args.download_only:
        return
    split_rows = {split: [row for row in rows if row["split"] == split]
                  for split in ("train", "dev", "test")}
    for split, selected in split_rows.items():
        save_npz(selected, out / f"{split}.npz", args.image_size, args.crop_rescale)
    audits = {
        split: overlay_audit(selected, split, out, args.image_size,
                             args.crop_rescale, args.seed)
        for split, selected in split_rows.items()
    }
    train_ids = {(row["source_split"], row["image_id"]) for row in split_rows["train"]}
    dev_ids = {(row["source_split"], row["image_id"]) for row in split_rows["dev"]}
    test_ids = {(row["source_split"], row["image_id"]) for row in split_rows["test"]}
    assert not train_ids & dev_ids and not train_ids & test_ids and not dev_ids & test_ids
    projection = np.asarray([row["source_projection_max_px"] for row in rows])
    assert np.isfinite(projection).all() and projection.max() < .25
    manifest = {
        "protocol": "rhd_full_no_kd_v1",
        "source": archive["url"],
        "download_source_override": bool(args.archive_url),
        "source_archive_size": archive["archive_size"],
        "license": "Research only; commercial use prohibited; cite Zimmermann and Brox, ICCV 2017.",
        "official_counts": {"training_images": 41258, "evaluation_images": 2728},
        "selection": {
            "scope": "all annotated left/right hand instances with positive 3D depth",
            "minimum_visible_joints": args.min_visible,
            "minimum_visible_extent_px": args.min_extent_px,
        },
        "splits": {
            split: {"instances": len(selected),
                    "unique_images": len({row["image_id"] for row in selected}),
                    "source_split": "training" if split == "train" else "evaluation"}
            for split, selected in split_rows.items()
        },
        "split_limit": "RHD has no subject/sequence identity. Official train/evaluation are isolated; dev/test use disjoint evaluation image IDs.",
        "supervision": {
            "2d": "21 joints with visibility",
            "3d": "21 metric camera-frame joints in meters; trained wrist-relative",
            "mano": "unavailable",
            "camera": "per-image pinhole K, used for source-label audit",
        },
        "canonicalization": "Left crops mirrored horizontally; left wrist-relative 3D x reflected. Joint order wrist then thumb/index/middle/ring/little base-to-tip.",
        "units": {"xyz": "meters", "uv": "pixels"},
        "projection_error_px": {
            "max": float(projection.max()), "mean": float(projection.mean()),
            "p95": float(np.percentile(projection, 95)),
        },
        "download": counters,
        "r0_audit": audits,
        "real_e2_exclusion": "No DexYCB input path or identifier is read by this script.",
        "sha256": {
            "archive_index": sha(index_path),
            "training_annotations": sha(DATA / "training/anno_training.pickle"),
            "evaluation_annotations": sha(DATA / "evaluation/anno_evaluation.pickle"),
            **{f"{split}_labels": sha(out / f"{split}.npz") for split in split_rows},
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "R0_passed", "splits": manifest["splits"],
                      "projection_error_px": manifest["projection_error_px"]}), flush=True)


if __name__ == "__main__":
    main()
