"""Fetch pinned repository archives, no execution of downloaded setup scripts."""
import hashlib
import json
import tarfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sources = {
    "Dyn-HaMR": ("ZhengdiYu/Dyn-HaMR", "fa9cd7412c205fd15ee4139c8caacf79bf6167e6"),
    "hamer": ("ZhengdiYu/hamer", "2a5b8ee852d87fdb56d65df7fb0e473c0d4f7648"),
    "Fast-HaMeR": ("hunainahmedj/Fast-HaMeR", "009821c80e9bfa4e111c6b9b641408126c53d03f"),
}
for name, (repo, commit) in sources.items():
    dst = ROOT / "repos" / name
    url = f"https://codeload.github.com/{repo}/tar.gz/{commit}"
    archive = ROOT / "repos" / (name + "-" + commit + ".tar.gz")
    record = {"component": name, "commit": commit, "url": url}
    start = time.perf_counter()
    try:
        if dst.exists():
            raise FileExistsError(f"{dst} exists; refusing replacement")
        if not archive.exists():
            partial = archive.with_suffix(".gz.partial")
            with urllib.request.urlopen(url, timeout=60) as resp, partial.open("xb") as out:
                while True:
                    block = resp.read(1024 * 1024)
                    if not block:
                        break
                    out.write(block)
                    if out.tell() > 1024**3:
                        raise ValueError("archive exceeded 1 GiB budget")
            partial.rename(archive)
        record["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
            prefix = members[0].name.split("/")[0]
            # Python managed 3.10 has data-filter backport; explicit safety checks too.
            for item in members:
                rel = Path(item.name)
                if rel.is_absolute() or ".." in rel.parts or item.issym() or item.islnk():
                    raise ValueError("unsafe archive member: " + item.name)
            tar.extractall(ROOT / "repos", filter="data")
        (ROOT / "repos" / prefix).rename(dst)
        record.update(status="ok", bytes=archive.stat().st_size)
    except Exception as exc:
        record.update(status="failed", error=repr(exc))
    record["elapsed_s"] = time.perf_counter()-start
    (ROOT / "artifacts/bootstrap" / (name+"-source.json")).write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps(record), flush=True)
