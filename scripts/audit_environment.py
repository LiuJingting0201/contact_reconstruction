#!/usr/bin/env python3
"""Bounded, read-only environment inventory and pinned public source retrieval."""
import argparse
import hashlib
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "Dyn-HaMR": {
        "repo": "ZhengdiYu/Dyn-HaMR",
        "commit": "fa9cd7412c205fd15ee4139c8caacf79bf6167e6",
        "files": [
            ".gitmodules", "README.md", "scripts/install_conda.sh",
            "dyn-hamr/data/dataset.py", "dyn-hamr/data/tools.py",
            "dyn-hamr/data/vidproc.py", "dyn-hamr/preproc/export_hamer.py",
            "dyn-hamr/preproc/launch_hamer.py", "dyn-hamr/preproc/extract_frames.py",
            "dyn-hamr/optim/base_scene.py", "dyn-hamr/optim/params.py",
            "dyn-hamr/geometry/camera.py", "dyn-hamr/optim/losses.py",
            "dyn-hamr/optim/optimizers.py", "dyn-hamr/confs/optim.yaml",
            "dyn-hamr/confs/config.yaml", "dyn-hamr/confs/data/video_vipe.yaml",
            "dyn-hamr/run_opt.py",
        ],
    },
    "hamer": {
        "repo": "ZhengdiYu/hamer",
        "commit": "2a5b8ee852d87fdb56d65df7fb0e473c0d4f7648",
        "files": ["run.py", "hamer/models/__init__.py", "hamer/models/hamer.py",
                  "hamer/configs/__init__.py", "hamer/datasets/vitdet_dataset.py"],
    },
    "Fast-HaMeR": {
        "repo": "hunainahmedj/Fast-HaMeR",
        "commit": "009821c80e9bfa4e111c6b9b641408126c53d03f",
        "files": ["train.py", "train_knowledge_dist.py", "hamer/models/hamer.py",
                  "hamer/models/efficient_hamer.py", "hamer/models/backbones/__init__.py",
                  "hamer/configs_hydra/experiment/resnet_50.yaml",
                  "hamer/configs_hydra/data/mix_all.yaml",
                  "hamer/configs/datasets_tar.yaml",
                  "hamer/configs_hydra/trainer/default_hamer.yaml"],
    },
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch-sources", action="store_true")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = ROOT / "artifacts" / "audit" / stamp
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "timestamp_utc": stamp, "argv": sys.argv, "cwd": str(Path.cwd()),
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "machine": platform.machine(),
        "script_sha256": sha(Path(__file__)),
        "modules": {n: importlib.util.find_spec(n) is not None
                    for n in ["numpy", "scipy", "torch", "matplotlib"]},
        "commands": {n: shutil.which(n) for n in ["uv", "nvidia-smi", "conda"]},
        "inventory_scope": "workspace inputs only; no scan of private home or remote hosts",
        "inputs": [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size,
                    "sha256": sha(p)} for p in sorted((ROOT / "inputs").rglob("*")) if p.is_file()],
        "source_documents": {str(p.relative_to(ROOT)): sha(p)
                             for p in sorted((ROOT / "docs/reference").glob("*.md"))},
    }
    if manifest["commands"]["nvidia-smi"]:
        proc = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                               "--format=csv"], text=True, capture_output=True, timeout=20)
        manifest["gpu_query"] = {"code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    write_json(out / "environment.json", manifest)
    events = []
    if args.fetch_sources:
        for name, spec in SOURCES.items():
            for rel in spec["files"]:
                url = f"https://raw.githubusercontent.com/{spec['repo']}/{spec['commit']}/{rel}"
                dst = ROOT / "vendor/pinned" / name / rel
                start = time.perf_counter()
                event = {"component": name, "commit": spec["commit"], "url": url,
                         "path": str(dst.relative_to(ROOT))}
                try:
                    with urllib.request.urlopen(url, timeout=25) as resp:
                        data = resp.read(5_000_001)
                    if len(data) > 5_000_000:
                        raise ValueError("source exceeds 5 MB per-file budget")
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if dst.exists() and dst.read_bytes() != data:
                        raise ValueError("existing pinned source differs; refusing overwrite")
                    if not dst.exists():
                        dst.write_bytes(data)
                    event.update(status="ok", bytes=len(data), sha256=sha(dst))
                except Exception as exc:
                    event.update(status="failed", error=repr(exc))
                event["elapsed_s"] = time.perf_counter() - start
                events.append(event)
                print(json.dumps(event), flush=True)
                write_json(out / "source_fetch.json", events)
    manifest["source_fetch_failures"] = sum(e["status"] == "failed" for e in events)
    write_json(out / "environment.json", manifest)
    print("AUDIT_DIRECTORY=" + str(out), flush=True)
    return int(manifest["source_fetch_failures"] > 0)


if __name__ == "__main__":
    sys.exit(main())
