"""Verify downloaded experiment files and recalculate selected metrics."""
import hashlib
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
path = root/"artifacts/remote/geometry/cpu_geom_v1_001"
expected = json.loads((path/"sha256.json").read_text())
hash_errors = [name for name, h in expected.items()
               if hashlib.sha256((path/name).read_bytes()).hexdigest() != h]
assert not hash_errors, hash_errors
rows = {r["case"]: r for r in json.loads((path/"results.json").read_text())}
arrays = np.load(path/"raw_arrays.npz")
for z in [.5, 1.]:
    values = arrays[f"noise_{z}_depth_m"]
    measured = np.std(values-z, ddof=1)*1000
    assert np.isclose(measured, rows[f"noise_depth_{z}"]["metrics"]["std_mm"], rtol=1e-12)
report = {"geometry_files_verified": len(expected), "hash_errors": hash_errors,
          "recomputed_metrics": ["noise_depth_0.5/std_mm", "noise_depth_1.0/std_mm"],
          "passed": True}
remote_root = root/"artifacts/remote"
remote_manifest = remote_root/"remote_sha256.json"
if remote_manifest.exists():
    remote_expected = json.loads(remote_manifest.read_text())["files"]
    errors = [name for name, h in remote_expected.items()
              if hashlib.sha256((remote_root/name).read_bytes()).hexdigest() != h]
    assert not errors, errors
    report["remote_evidence_files_verified"] = len(remote_expected)
report["rhd_pilots"] = {}
for run_id in ["rhd_supervised_v2_run02","rhd_supervised_v3"]:
    pilot = remote_root/run_id
    if not (pilot/"results.json").exists():
        continue
    result = json.loads((pilot/"results.json").read_text())
    for relative in ["scripts/train_rhd_supervised.py",f'experiments/{result["settings"]["run_id"].split("_run")[0]}.json',"inputs/rhd/manifest.json"]:
        snapshot = pilot/Path(relative).name
        assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == result["source_sha256"][relative]
    steps = [json.loads(line) for line in (pilot/"steps.jsonl").read_text().splitlines()]
    settings = result["settings"]
    assert [row["step"] for row in steps] == list(range(1,settings["steps"]+1))
    ids = set(np.load(pilot/"training_targets.npz")["ids"].tolist())
    for row in steps:
        assert len(row["sample_ids"]) == settings["batch_size"]
        assert set(row["sample_ids"]) <= ids
        assert np.isfinite([row["loss"],row["grad_norm_before_clip"],*row["terms"].values()]).all()
        assert np.isclose(row["loss"],sum(row["terms"][k]*v for k,v in settings["weights"].items()),rtol=1e-6)
    recomputed = []
    for line in (pilot/"evaluations.jsonl").read_text().splitlines():
        evaluation = json.loads(line)
        for split, recorded in evaluation["splits"].items():
            target = np.load(pilot/f"{split}_targets.npz")
            pred = np.load(pilot/f'{split}_predictions_{evaluation["step"]:04d}.npz')
            assert np.array_equal(target["ids"],pred["ids"])
            e3 = np.linalg.norm(pred["xyz_relative_m"]-target["xyz_relative_m"],axis=-1)*1000
            e2 = np.linalg.norm(pred["uv_normalized"]-target["uv_normalized"],axis=-1)*settings["image_size"]
            assert np.allclose(e3,pred["error3d_mm"],atol=1e-5)
            assert np.allclose(e2,pred["error2d_px"],atol=1e-5)
            assert np.isclose(e3.mean(),recorded["mpjpe_mm"],rtol=1e-6)
            assert np.isclose(e2[target["visible"]].mean(),recorded["visible_crop_error_px"],rtol=1e-6)
            if "predicted_bbox_span_px_min_median_max" in recorded:
                span = np.quantile(np.ptp(pred["uv_normalized"],axis=1).max(1)*256,[0,.5,1])
                assert np.allclose(span,recorded["predicted_bbox_span_px_min_median_max"])
            recomputed.append(f'{split}/step_{evaluation["step"]}')
    report["rhd_pilots"][run_id] = {"steps_verified":len(steps),"evaluations_recomputed":recomputed}
if len(report["rhd_pilots"]) == 2:
    orders = [[json.loads(line)["sample_ids"] for line in (remote_root/run/"steps.jsonl").read_text().splitlines()]
              for run in report["rhd_pilots"]]
    assert orders[0] == orders[1]
    report["rhd_same_batch_order_verified"] = True
(root/"artifacts/verification.json").write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(report, indent=2))
