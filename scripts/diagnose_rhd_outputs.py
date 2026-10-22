"""Post-hoc diagnostics from saved arrays, without retraining or GT leakage."""
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
report = {}
for run in ["rhd_supervised_v2_run02", "rhd_supervised_v3"]:
    path = root/"artifacts/remote"/run
    train = np.load(path/"training_targets.npz")
    target = np.load(path/"evaluation_targets.npz")
    prediction = np.load(path/"evaluation_predictions_0200.npz")
    pred, gt = prediction["xyz_relative_m"], target["xyz_relative_m"]
    train_pred = np.load(path/"training_predictions_0200.npz")["xyz_relative_m"]
    u,_,vt = np.linalg.svd(train_pred.reshape(-1,3).T@train["xyz_relative_m"].reshape(-1,3))
    fix = np.eye(3)
    fix[-1,-1] = np.linalg.det(u@vt)
    common_rotation = u@fix@vt
    mean_train = train["xyz_relative_m"].mean(0)
    mpjpe = lambda x,y: float(np.linalg.norm(x-y,axis=-1).mean()*1000)
    rotation_errors, camera_errors, oracle_scales = [], [], []
    for x,y,uv,mask in zip(pred,gt,target["uv_normalized"],target["visible"]):
        # Rotation about the wrist only; no translation, reflection, or scale.
        u,_,vt = np.linalg.svd(x.T@y)
        correction = np.eye(3)
        correction[-1,-1] = np.linalg.det(u@vt)
        rotation_errors.append(mpjpe(x@(u@correction@vt),y))
        # Diagnostic oracle using evaluation GT: shared positive scale + 2D offset.
        # Not a deployable prediction or a replacement evaluation metric.
        xy = x[mask,:2].astype(np.float64)
        q = uv[mask].astype(np.float64)
        centered_x, centered_q = xy-xy.mean(0), q-q.mean(0)
        scale = max(0., float((centered_x*centered_q).sum()/(centered_x**2).sum()))
        fit = scale*centered_x+q.mean(0)
        camera_errors.extend((np.linalg.norm(fit-q,axis=-1)*256).tolist())
        oracle_scales.append(scale*2)
    rows = [json.loads(line) for line in (path/"steps.jsonl").read_text().splitlines()]
    settings = json.loads((path/"results.json").read_text())["settings"]
    weighted = {k:float(np.mean([s["terms"][k]*w for s in rows[-50:]]))
                for k,w in settings["weights"].items()}
    report[run] = {
        "actual_mpjpe_mm":mpjpe(pred,gt),
        "one_common_rotation_fit_on_training_eval_mpjpe_mm":mpjpe(pred@common_rotation,gt),
        "common_rotation_angle_degrees":float(np.degrees(np.arccos(np.clip((np.trace(common_rotation)-1)/2,-1,1)))),
        "train_mean_pose_baseline_mpjpe_mm":mpjpe(mean_train[None],gt),
        "all_joints_at_wrist_baseline_mpjpe_mm":mpjpe(np.zeros_like(gt),gt),
        "GT_oracle_wrist_rotation_only_mpjpe_mm":float(np.mean(rotation_errors)),
        "GT_oracle_positive_weak_camera_visible_2d_px":float(np.mean(camera_errors)),
        "GT_oracle_weak_scale_quantiles":np.quantile(oracle_scales,[0,.5,1]).tolist(),
        "last50_mean_weighted_loss_terms":weighted,
        "scope":"Post-hoc diagnostics. GT oracle fits are not model performance. MSE-fitted camera is scored with mean Euclidean error, so it is not a strict lower bound for that metric."}
out = root/"artifacts/rhd_output_diagnosis.json"
out.write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
