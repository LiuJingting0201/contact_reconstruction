"""Collect and verify the existing M-S1 runs; never launch training or inference."""
import fcntl
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "resnet50_real_mixed_m_s1_subject02_20261021_01"
CORRECTION = "resnet50_real_mixed_m_s1_sampler_correction_20261021_01"
REMOTE = "simready-dev:/workspace/pose_reconstract/"
STATE = ROOT / "logs/m_s1_collection_state.json"
PYTHON = ROOT / ".venv-dex-download/bin/python"
COMPARISON = ROOT / "artifacts/m_s1_final_comparison_20261022"


def status(stage, **kwargs):
    value = {"stage": stage, "updated_unix": time.time(), **kwargs}
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(STATE)
    print(json.dumps(value), flush=True)


def run(args):
    subprocess.run([str(a) for a in args], cwd=ROOT, check=True)


def main():
    lock = (ROOT / "logs/m_s1_collection.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("An M-S1 collector is already running.", flush=True)
        return
    remote_code = f"""
import json
from pathlib import Path
p=Path('/workspace/pose_reconstract/artifacts/{CORRECTION}')
result={{}}
for name in ('selection_state.json','completion.json','verification.json'):
    f=p/name
    if f.exists():
        data=json.loads(f.read_text())
        result[name]={{k:data[k] for k in ('step','best_step','best_mpjpe','passed',
            'stop_reason','last_step') if k in data}}
f=p/'steps.jsonl'
if f.exists():
    with f.open('rb') as stream:
        stream.seek(max(0,f.stat().st_size-16384))
        lines=stream.read().splitlines()
    for line in reversed(lines):
        try:
            data=json.loads(line)
            result['latest_update']={{k:data.get(k) for k in
                ('step','elapsed_s','loss','projection_collapse_rate')}}
            break
        except (ValueError,UnicodeDecodeError):
            continue
print(json.dumps(result))
"""
    deadline = time.monotonic() + 3 * 3600
    while time.monotonic() < deadline:
        try:
            output = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=15", "simready-dev",
                 "/workspace/pose_reconstract/.venv-gpu/bin/python -"],
                input=remote_code, text=True, capture_output=True, timeout=45, check=True)
            remote_state = json.loads(output.stdout)
            status("waiting_remote_verification", remote=remote_state)
            if remote_state.get("verification.json", {}).get("passed"):
                break
        except (subprocess.SubprocessError, ValueError) as error:
            status("remote_status_retry", error=str(error))
        time.sleep(45)
    else:
        raise TimeoutError("Remote verification was not ready within three hours.")
    directories = [
        SOURCE, CORRECTION,
        "m_s1_frozen_splits_20261021_01",
        "m_s1_coordinate_audit_20261021_01",
        "m_s1_frozen_splits_sampler_correction_20261021_01",
        "m_s1_coordinate_audit_sampler_correction_20261021_01",
    ]
    for name in directories:
        status("syncing_artifacts", directory=name)
        run(["rsync", "-a", "--partial", REMOTE + f"artifacts/{name}/",
             f"artifacts/{name}/"])
    for name in (SOURCE, CORRECTION):
        run(["rsync", "-a", "--partial", REMOTE + f"logs/{name}_pipeline.log", "logs/"])
        status("verifying_local_artifacts", run=name)
        run([PYTHON, "scripts/verify_m_s1_artifacts.py", "--run", f"artifacts/{name}"])
    status("writing_final_comparison")
    run([PYTHON, "scripts/compare_m_s1_runs.py",
         "--source", f"artifacts/{SOURCE}", "--correction", f"artifacts/{CORRECTION}",
         "--output", COMPARISON])
    comparison = json.loads((COMPARISON / "comparison.json").read_text())
    report = json.loads((ROOT / "artifacts" / CORRECTION / "results.json").read_text())
    completion = report["completion"]
    record = ROOT / "docs/dyn_hamr_experiment_record.md"
    heading = "### Completed M-S1 sampler correction and final comparison"
    if heading not in record.read_text():
        text = [
            "", heading, "",
            f"`{CORRECTION}` stopped with `{completion['stop_reason']}` at "
            f"{completion['last_step']} updates; frozen best-dev is update "
            f"{completion['best_step']}. Both runs passed local saved-artifact verification, "
            "including checkpoint hashes and ordered full-dev sample IDs.", "",
            "| Independent dev | Wrist MPJPE mm | Fingertip mm | Visible 2D px | Failure |",
            "|---|---:|---:|---:|---:|",
        ]
        for domain in ("dexycb", "freihand", "rhd"):
            m = report["best"]["domains"][domain]
            text.append(f"| {domain} | {m['wrist_relative_mpjpe_mm']:.3f} | "
                        f"{m['fingertip_wrist_relative_error_mm']:.3f} | "
                        f"{m['visible_2d_epe_px']:.3f} | {m['failure_rate']:.4%} |")
        gain = comparison["relative_improvement_percent"]["correction"]["dexycb"]
        text += [
            "", "Same-dev DexYCB improvement over the frozen RHD-only checkpoint: "
            f"wrist {gain['wrist_relative_mpjpe_mm']:.2f}%, "
            f"fingertip {gain['fingertip_wrist_relative_error_mm']:.2f}%, "
            f"2D {gain['visible_2d_epe_px']:.2f}%.",
            f"M-S2 eligible: **{report['m_s2_eligible']}**. Failed gates: "
            + ", ".join(k for k, v in report["gate_flags"].items() if not v) + ".",
            "Remaining severe domain regression: " + (", ".join(
                f"{r['domain']}/{r['metric']}" for r in report["domain_imbalance"])
                or "none") + ".",
            "Training collapse records: `" + json.dumps(report["train_collapse_summary"]) + "`.",
            "", *[f"- {name} SHA256: `{digest}`"
                  for name, digest in completion["checkpoint_hashes"].items()],
            "", "The comparison and descriptive paired 2D intervals are in "
            "`artifacts/m_s1_final_comparison_20261022/comparison.md` and `comparison.json`. "
            "Per-run reports retain all domain metrics, train/dev curves, masks, "
            "data inventories, exposure counts, early failures and geometry intervals.",
            "The dev set contains one independent DexYCB subject; the sequence intervals "
            "do not establish cross-subject generalization. The correction uses the same "
            "dev for selection, so its comparison is descriptive rather than a final test.",
            "Stopped after this single M-S1 correction. No third run, M-S2, E3, Final-E4 "
            "inference, KD, B2 tuning, Dyn-HaMR integration or retargeting was run.", "",
        ]
        with record.open("a") as stream:
            stream.write("\n".join(text) + "\n")
    status("complete", runs=[SOURCE, CORRECTION], comparison=str(COMPARISON),
           m_s2_eligible=report["m_s2_eligible"], further_training_started=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        status("failed", error=repr(error))
        raise
