"""Regression check: an all-nonfinite later dev must still produce a failed report."""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import train_resnet50_m_s1 as ms


def main():
    rng = np.random.default_rng(7)
    target = rng.normal(size=(4, 21, 3)) * .02
    target -= target[:, :1]
    arrays = {
        "xyz": target * 1.5, "target_xyz": target,
        "uv": rng.normal(size=(4, 21, 2)) * .1,
        "target_uv": np.zeros((4, 21, 2)), "valid": np.ones((4, 21), bool),
        "cam": np.ones((4, 3)), "depth": np.ones((4, 21)),
        "orient": np.full(4, np.nan),
    }
    baseline = ms.metrics(arrays)
    newer = {**arrays, "xyz": target * 1.2}
    better = ms.metrics(newer)
    failed = ms.metrics({**arrays, "xyz": np.full_like(target, np.nan)})
    with tempfile.TemporaryDirectory(prefix="m_s1_report_check_") as name:
        out = Path(name)
        shutil.copy2(ms.ROOT / "configs/resnet50_real_mixed_m_s1_subject02.json",
                     out / "resnet50_real_mixed_m_s1_subject02.json")
        rows = []
        for tag, metric in (("rhd_only_baseline", baseline), ("dev_0000000", baseline),
                            ("dev_0001000", better), ("dev_0002000", failed)):
            rows.append({"tag": tag, "domains": {d: metric for d in ms.DOMAINS}})
        (out / "evaluations.jsonl").write_text("\n".join(json.dumps(x) for x in rows))
        (out / "steps.jsonl").write_text(json.dumps({"step": 1, "loss": 1.}) + "\n")
        (out / "completion.json").write_text(json.dumps({
            "best_step": 1000, "last_step": 2000, "stop_reason": "nonfinite_dev",
            "checkpoint_hashes": {},
        }))
        ids = np.array(["subject/sequence1:000001:cam", "subject/sequence1:000002:cam",
                        "subject/sequence2:000001:cam", "subject/sequence2:000002:cam"])
        np.savez(out / "rhd_only_baseline_dexycb_predictions.npz", sample_id=ids, **arrays)
        np.savez(out / "dev_0001000_dexycb_predictions.npz", sample_id=ids, **newer)
        subprocess.run([sys.executable, str(ms.ROOT / "scripts/report_m_s1.py"),
                        "--run", str(out)], check=True)
        result = json.loads((out / "results.json").read_text())
        assert not result["m_s2_eligible"]
        assert not result["gate_flags"]["real_dev_last_three_evaluations_improving"]
        assert len(result["projection_history"]) == 4
        assert (out / "results.md").exists()
    print("PASS: all-nonfinite later dev is retained and reported as ineligible.")


if __name__ == "__main__":
    main()
