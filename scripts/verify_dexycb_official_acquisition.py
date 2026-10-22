"""Verify complete official archive, selected extraction and download provenance."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile
import time

SUBJECT_URL = "https://drive.google.com/file/d/1Ehh92wDE3CWAiKG7E9E73HjN2Xk2XfEk"
CALIBRATION_URL = "https://drive.google.com/file/d/1UAwVKT4Rgb1fLcFoa1o71_-0NtSvvLAQ"
EXPECTED_SIZE = 12412314463


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--subset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if a.archive.stat().st_size != EXPECTED_SIZE:
        raise ValueError(f"incomplete archive: {a.archive.stat().st_size} != {EXPECTED_SIZE}")
    start = time.time()
    sha = digest(a.archive)
    print(json.dumps({"stage": "sha256", "sha256": sha}), flush=True)
    uncompressed = 0
    with gzip.open(a.archive, "rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            uncompressed += len(block)
    print(json.dumps({"stage": "gzip_crc_eof", "uncompressed_bytes": uncompressed}), flush=True)
    selected = {str(f.relative_to(a.subset)): digest(f) for f in a.subset.rglob("*")
                if f.is_file() and (f.parts[-1] != "extraction.json")}
    verified = set()
    member_count, sequences = 0, set()
    # Validate bytes actually extracted before the full download finished.
    # Stop subject traversal once all selected subject files have been matched;
    # gzip CRC for the entire archive was independently checked above.
    desired = {k for k in selected if not k.startswith("calibration/")}
    with tarfile.open(a.archive, "r|gz") as archive:
        for member in archive:
            member_count += 1
            parts = Path(member.name).parts
            if len(parts) >= 2:
                sequences.add("/".join(parts[:2]))
            if member.isfile() and member.name in desired:
                stream = archive.extractfile(member)
                h = hashlib.sha256()
                for block in iter(lambda: stream.read(1024*1024), b""):
                    h.update(block)
                if h.hexdigest() != selected[member.name]:
                    raise ValueError(f"extraction mismatch {member.name}")
                verified.add(member.name)
            if verified == desired:
                break
    if verified != desired:
        raise ValueError(f"missing selected archive entries: {desired-verified}")
    with tarfile.open(a.calibration, "r:gz") as archive:
        for member in archive:
            if member.isfile() and member.name in selected:
                h = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
                if h != selected[member.name]:
                    raise ValueError(f"calibration extraction mismatch {member.name}")
                verified.add(member.name)
    if verified != set(selected):
        raise ValueError("not all selected inputs verified")
    report = {"subject": {"official_url": SUBJECT_URL, "bytes": a.archive.stat().st_size,
                          "sha256": sha, "gzip_crc_eof_passed": True, "uncompressed_bytes": uncompressed},
              "calibration": {"official_url": CALIBRATION_URL, "bytes": a.calibration.stat().st_size,
                              "sha256": digest(a.calibration)},
              "selected_input_sha256": selected, "selected_files_verified": len(verified),
              "subject_members_scanned_for_subset": member_count, "elapsed_s": time.time()-start,
              "hash_scope": "computed SHA256; official source did not publish a comparison SHA256",
              "download_result": "complete official archive; public normal large-file confirmation; no login bypass"}
    a.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"passed": True, "selected_files_verified": len(verified)}), flush=True)


if __name__ == "__main__":
    main()
