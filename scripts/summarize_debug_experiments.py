"""Recompute debug experiment metrics and compare preserved pre/post arrays."""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--remote-copy",action="store_true")
args = parser.parse_args()
base = ROOT/"artifacts"/"remote" if args.remote_copy else ROOT/"artifacts"
report = {"reproductions":{},"runs":{}}
for original,reproduced in [("rhd_supervised_v2_run02","rhd_debug_pre_v2"),
                            ("rhd_supervised_v3","rhd_debug_pre_v3")]:
    differences = {}
    for step in [0,50,100,150,200]:
        for split in ["training","evaluation"]:
            a = np.load(base/original/f"{split}_predictions_{step:04d}.npz")
            b = np.load(base/reproduced/f"{split}_predictions_{step:04d}.npz")
            for key in ["xyz_relative_m","uv_normalized"]:
                difference = float(np.max(np.abs(a[key]-b[key])))
                differences[f"{split}_{step}_{key}"] = difference
                if step == 0:
                    assert difference < 1e-6, (original,step,key,difference)
    report["reproductions"][reproduced] = {"max_array_difference":max(differences.values()),
                                         "all_arrays_match_1e_6":all(v < 1e-6 for v in differences.values()),
                                         "details":differences}

for run in ["rhd_supervised_v2_run02","rhd_supervised_v3","rhd_supervised_v4",
            "rhd_supervised_v5","rhd_supervised_v5_long","rhd_v4_deterministic_a",
            "rhd_v4_deterministic_b","rhd_v5_deterministic"]:
    directory = base/run
    result = json.loads((directory/"results.json").read_text())
    settings = result["settings"]
    steps = [json.loads(x) for x in (directory/"steps.jsonl").read_text().splitlines()]
    assert len(steps) == settings["steps"]
    assert [x["step"] for x in steps] == list(range(1,len(steps)+1))
    for row in steps:
        assert np.isfinite([row["loss"],row["grad_norm_before_clip"],*row["terms"].values()]).all()
        assert np.isclose(row["loss"],sum(row["terms"][k]*v for k,v in settings["weights"].items()),rtol=1e-6)
    evaluations = [json.loads(x) for x in (directory/"evaluations.jsonl").read_text().splitlines()]
    for evaluation in evaluations:
        for split,metrics in evaluation["splits"].items():
            p = np.load(directory/f'{split}_predictions_{evaluation["step"]:04d}.npz')
            t = np.load(directory/f"{split}_targets.npz")
            assert np.array_equal(p["ids"],t["ids"])
            e3 = np.linalg.norm(p["xyz_relative_m"]-t["xyz_relative_m"],axis=-1)*1000
            e2 = np.linalg.norm(p["uv_normalized"]-t["uv_normalized"],axis=-1)*256
            assert np.isclose(e3.mean(),metrics["mpjpe_mm"],rtol=1e-6)
            assert np.isclose(e2[t["visible"]].mean(),metrics["visible_crop_error_px"],rtol=1e-6)
            if settings.get("positive_camera"):
                assert (p["weak_camera"][:,0] > .05).all()
                assert (p["camera_space_joint_depth_m"] > 0).all()
    if settings.get("positive_camera"):
        assert result["max_backbone_parameter_update"] > 0
        assert result["reload_uv_max_diff_normalized"] < 1e-6
    report["runs"][run] = {"updates":len(steps),"final":result["final"]["splits"],
                          "reload_joint_difference_m":result["reload_joint_max_diff_m"],
                          "wall_s":result["wall_s_including_evaluation_save_reload"]}
orders = [[json.loads(x)["sample_ids"] for x in (base/run/"steps.jsonl").read_text().splitlines()][:200]
          for run in report["runs"]]
assert all(x == orders[0] for x in orders)
report["same_first_200_batches"] = True
deterministic_differences = []
for step in [0,50,100,150,200]:
    for split in ["training","evaluation"]:
        a = np.load(base/"rhd_v4_deterministic_a"/f"{split}_predictions_{step:04d}.npz")
        b = np.load(base/"rhd_v4_deterministic_b"/f"{split}_predictions_{step:04d}.npz")
        for key in ["xyz_relative_m","uv_normalized","weak_camera"]:
            deterministic_differences.append(float(np.max(np.abs(a[key]-b[key]))))
assert max(deterministic_differences) < 1e-6
report["deterministic_repeat_max_difference"] = max(deterministic_differences)
report["mano_controls"] = json.loads((base/"mano_debug_controls/results.json").read_text())
teacher = json.loads((base/"teacher_geometry_control_run02/results.json").read_text())
teacher_arrays = np.load(base/"teacher_geometry_control_run02/outputs.npz")
target = np.load(base/"rhd_supervised_v3/evaluation_targets.npz")
errors = []
for i in range(len(target["ids"])):
    uv = teacher_arrays[f"{i}_nominal_pred_keypoints_2d"][0]
    e = np.linalg.norm(uv-target["uv_normalized"][i],axis=-1)*256
    errors.extend(e[target["visible"][i]].tolist())
    xyz = teacher_arrays[f"{i}_nominal_pred_keypoints_3d"][0]
    visible = target["visible"][i]
    gt_crop = (target["uv_normalized"][i]+.5)*256
    affine3 = np.vstack([target["affine"][i],[0,0,1]])
    gt_full = (np.c_[gt_crop,np.ones(21)]@np.linalg.inv(affine3).T)[:,:2]
    for key in ["global_orient","hand_pose","betas","pred_cam","pred_keypoints_3d"]:
        assert np.allclose(teacher_arrays[f"{i}_nominal_{key}"],teacher_arrays[f"{i}_real_{key}"],atol=1e-6,rtol=0)
    for state in ["initial","pseudo_fit","gt_oracle"]:
        translation = teacher_arrays[f"{i}_{state}_translation_m"]
        camera = xyz+translation
        assert (camera[:,2] > 0).all()
        homogeneous = camera@target["K"][i].T
        uv_full = homogeneous[:,:2]/homogeneous[:,2:]
        assert np.allclose(uv_full,teacher_arrays[f"{i}_{state}_uv_full"],atol=1e-6)
        mean_error = np.linalg.norm(uv_full-gt_full,axis=-1)[visible].mean()
        # GT crop uses float32 affine/normalization, allow its subpixel roundoff.
        assert np.isclose(mean_error,teacher["rows"][i][state]["visible_full_error_px"],atol=1e-4)
        depth_error = abs(camera[0,2]-target["xyz_absolute_m"][i,0,2])*1000
        assert np.isclose(depth_error,teacher["rows"][i][state]["wrist_depth_abs_error_mm"],atol=1e-4)
report["teacher"] = teacher["mean_of_per_image_metrics"]
report["teacher"]["pooled_visible_crop_error_px"] = float(np.mean(errors))
report["teacher"]["all_fits_success"] = all(all(r["fits"].values()) for r in teacher["rows"])
destination = ROOT/"artifacts/debug_experiment_summary.json" if args.remote_copy else base/"debug_experiment_summary.json"
destination.write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({k:v for k,v in report.items() if k not in ("runs","reproductions")},indent=2))
for run,record in report["runs"].items():
    print(run,json.dumps({s:{k:v for k,v in m.items() if not k.startswith("per_image")}
                           for s,m in record["final"].items()}))
