"""Select first N complete archive sequences without extracting whole subject.

The gzip may still be downloading; selected sequences must finish before the next
sequence header. Whole archive SHA256/integrity is verified separately at completion.
"""
import argparse
import json
from pathlib import Path
import tarfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sequences", type=int, default=3)
    p.add_argument("--calibration", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    chosen, files = [], 0
    with tarfile.open(a.calibration, "r:gz") as archive:
        for member in archive:
            target = (a.output/member.name).resolve()
            if not target.is_relative_to(a.output.resolve()) or member.issym() or member.islnk():
                raise ValueError("unsafe calibration archive member")
            archive.extract(member, a.output)
    with tarfile.open(a.archive, "r|gz") as archive:
        for member in archive:
            parts = Path(member.name).parts
            if len(parts) < 2:
                continue
            sequence = "/".join(parts[:2])
            if sequence not in chosen:
                if len(chosen) >= a.sequences:
                    break
                chosen.append(sequence)
            target = (a.output/member.name).resolve()
            if not target.is_relative_to(a.output.resolve()) or member.issym() or member.islnk():
                raise ValueError("unsafe subject archive member")
            archive.extract(member, a.output)
            files += int(member.isfile())
    (a.output/"extraction.json").write_text(json.dumps({
        "source": str(a.archive.resolve()), "sequences": chosen, "files": files,
        "selection": "first complete sequences in official archive order, before model inference",
        "full_archive_integrity": "pending separate final archive verification"
    }, indent=2)+"\n")
    print(json.dumps({"sequences": chosen, "files": files}), flush=True)


if __name__ == "__main__":
    main()
