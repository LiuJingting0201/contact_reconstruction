"""Extract a deterministic supervised pilot subset from the official RHD ZIP."""
import argparse
import hashlib
import json
import pickle
import struct
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INDEX = json.loads((ROOT/"artifacts/rhd_archive_index.json").read_text())
ENTRIES = {x["name"]: x for x in INDEX["entries"]}
DEST = ROOT/"inputs/rhd"


def fetch(relative):
    entry = ENTRIES["RHD_published_v2/"+relative]
    path = DEST/relative
    if path.exists() and zlib.crc32(path.read_bytes()) == entry["crc"]:
        return path
    start = entry["offset"]
    end = start+30+len(entry["name"].encode())+65535+entry["compressed"]-1
    req = urllib.request.Request(INDEX["url"], headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=120) as response:
        assert response.status == 206
        assert response.headers["Content-Range"].startswith(f"bytes {start}-")
        blob = response.read()
    fields = struct.unpack("<IHHHHHIIIHH", blob[:30])
    assert fields[0] == 0x04034B50 and fields[3] in (0, 8)
    offset = 30+fields[-2]+fields[-1]
    data = blob[offset:offset+entry["compressed"]]
    if fields[3] == 8:
        data = zlib.decompress(data, -15)
    assert len(data) == entry["bytes"] and zlib.crc32(data) == entry["crc"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    print(json.dumps({"fetched":relative,"bytes":len(data),"crc32":entry["crc"]}),flush=True)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--config", default="experiments/rhd_supervised_v2.json")
    args = parser.parse_args()
    for name in ["README", "view_samples.py"]:
        fetch(name)
    if args.metadata_only:
        return
    settings = json.loads((ROOT/args.config).read_text())
    manifest = {"url": INDEX["url"], "settings": settings, "splits": {}}
    # Official RHD: left 0..20, right 21..41; wrist then tip-to-base fingers.
    order = np.array(settings["rhd_to_mano_order"])
    for split, count in settings["subset_counts"].items():
        anno_path = fetch(f"{split}/anno_{split}.pickle")
        # Only load the dataset author's published annotation, after ZIP CRC verification.
        annotations = pickle.loads(anno_path.read_bytes())
        candidates = []
        for key in sorted(annotations):
            a = annotations[key]
            uv = a["uv_vis"][21:42][order].astype(np.float32)
            xyz = a["xyz"][21:42][order].astype(np.float32)
            inside = ((uv[:,:2]>=0)&(uv[:,:2]<320)).all(1)
            extent = np.ptp(uv[:,:2],axis=0).max()
            if (uv[:,2]>0).sum() >= settings["min_visible"] and inside.all() and extent>=settings["min_extent_px"] and (xyz[:,2]>0).all():
                candidates.append(int(key))
        rng = np.random.default_rng(settings["seed"])
        selected = sorted(rng.choice(candidates,count,replace=False).tolist())
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(fetch, [f"{split}/color/{key:05d}.png" for key in selected]))
        rows, xyzs, uvs, ks = [], [], [], []
        for key in selected:
            a = annotations[key]
            uv, xyz = a["uv_vis"][21:42][order], a["xyz"][21:42][order]
            projected = xyz @ a["K"].T
            error = float(np.linalg.norm(projected[:,:2]/projected[:,2:]-uv[:,:2],axis=1).max())
            if error > settings["max_projection_error_px"]:
                raise ValueError(f"{split}/{key}: K/xyz/uv mismatch {error}")
            image = DEST/f"{split}/color/{key:05d}.png"
            rows.append({"id":key,"image":str(image.relative_to(DEST)),
                         "sha256":hashlib.sha256(image.read_bytes()).hexdigest(),
                         "max_projection_error_px":error})
            xyzs.append(xyz)
            uvs.append(uv)
            ks.append(a["K"])
        labels = DEST/f"{split}_subset.npz"
        np.savez_compressed(labels,ids=selected,xyz=np.array(xyzs),uv_vis=np.array(uvs),K=np.array(ks))
        manifest["splits"][split] = {"eligible_images":len(candidates),"rows":rows,
            "annotation_sha256":hashlib.sha256(anno_path.read_bytes()).hexdigest(),
            "labels_sha256":hashlib.sha256(labels.read_bytes()).hexdigest()}
    (DEST/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({s:{"selected":len(v["rows"]),"eligible":v["eligible_images"]} for s,v in manifest["splits"].items()}))


if __name__ == "__main__":
    main()
