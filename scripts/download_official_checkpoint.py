"""Resumable parallel ranges from the author's public HaMeR demo.

No credentials, mirrors or untrusted model sources. Range responses and final
LFS SHA256 are checked. Download events contain no signed redirect URLs.
"""
import concurrent.futures
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parents[1]
meta = json.loads((root/"artifacts/official_hamer_space.json").read_text())
revision = meta["sha"]
api = f"https://huggingface.co/api/spaces/geopavlakos/HaMeR/tree/{revision}/_DATA/hamer_ckpts/checkpoints"
with urllib.request.urlopen(api, timeout=30) as r:
    files = json.load(r)
entry = next(e for e in files if e["path"].endswith("/hamer.ckpt"))
size = entry["size"]
expected = entry["lfs"]["oid"]
url = f"https://huggingface.co/spaces/geopavlakos/HaMeR/resolve/{revision}/{entry['path']}"
out = root/"inputs/official_hamer"/entry["path"]
out.parent.mkdir(parents=True, exist_ok=True)
parts = out.parent/"download_parts"
parts.mkdir(exist_ok=True)
record_path = root/"artifacts/checkpoint_download.json"
record = {"repository": meta["repo"], "revision": revision, "path": entry["path"],
          "bytes": size, "expected_sha256": expected, "status": "downloading"}
record_path.write_text(json.dumps(record, indent=2)+"\n")
chunk = 64*1024*1024

def download(index):
    begin, end = index*chunk, min(size-1, (index+1)*chunk-1)
    dst = parts/f"{index:04d}.part"
    if dst.exists() and dst.stat().st_size == end-begin+1:
        return index
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={begin}-{end}"})
            with urllib.request.urlopen(req, timeout=45) as response, dst.open("wb") as stream:
                if response.status != 206 or response.headers.get("Content-Range") != f"bytes {begin}-{end}/{size}":
                    raise ValueError("server did not honor range")
                while True:
                    block = response.read(1024*1024)
                    if not block:
                        break
                    stream.write(block)
            if dst.stat().st_size != end-begin+1:
                raise ValueError("incomplete range")
            print(json.dumps({"chunk": index, "bytes": dst.stat().st_size, "status":"ok"}), flush=True)
            return index
        except Exception as exc:
            # URL exceptions may include signed locations; output type only.
            print(json.dumps({"chunk":index, "attempt":attempt, "error_type":type(exc).__name__}), flush=True)
            if attempt == 2:
                raise

try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(download, range((size+chunk-1)//chunk)))
    digest = hashlib.sha256()
    partial = out.with_suffix(".ckpt.partial")
    with partial.open("wb") as stream:
        for p in sorted(parts.glob("*.part")):
            with p.open("rb") as src:
                while True:
                    block = src.read(1024*1024)
                    if not block:
                        break
                    digest.update(block); stream.write(block)
    if digest.hexdigest() != expected or partial.stat().st_size != size:
        raise ValueError("checkpoint hash or size mismatch")
    os.replace(partial, out)
    record.update(status="complete", sha256=digest.hexdigest())
except Exception as exc:
    record.update(status="failed", error_type=type(exc).__name__)
    raise
finally:
    record_path.write_text(json.dumps(record, indent=2)+"\n")
