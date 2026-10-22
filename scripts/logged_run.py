#!/usr/bin/env python3
"""Capture a command without logging environment variables or using a shell."""
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[1]
command = sys.argv[1:]
if not command:
    raise SystemExit("usage: python3 scripts/logged_run.py COMMAND [ARG ...]")
out = root / "artifacts/commands" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
out.mkdir(parents=True, exist_ok=False)
record = {"argv": command, "cwd": str(Path.cwd()), "start_utc": datetime.now(timezone.utc).isoformat()}
(out / "command.json").write_text(json.dumps(record, indent=2) + "\n")
print("COMMAND_DIRECTORY=" + str(out), flush=True)
start = time.perf_counter()
with (out / "stdout.log").open("w") as stdout, (out / "stderr.log").open("w") as stderr:
    proc = subprocess.run(command, stdout=stdout, stderr=stderr)
record.update(returncode=proc.returncode, elapsed_s=time.perf_counter() - start,
              end_utc=datetime.now(timezone.utc).isoformat())
record["logs_sha256"] = {n: hashlib.sha256((out / n).read_bytes()).hexdigest()
                         for n in ["stdout.log", "stderr.log"]}
(out / "command.json").write_text(json.dumps(record, indent=2) + "\n")
print((out / "stdout.log").read_text(), end="")
print((out / "stderr.log").read_text(), end="", file=sys.stderr)
sys.exit(proc.returncode)
