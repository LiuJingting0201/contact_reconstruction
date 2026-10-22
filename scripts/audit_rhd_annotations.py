"""Diagnose official XYZ/K/UV residuals before changing pilot acceptance."""
import json
import pickle
import numpy as np
from fetch_rhd_subset import ROOT, fetch

report = {}
settings = json.loads((ROOT/"experiments/rhd_supervised_v1.json").read_text())
for split in ["training","evaluation"]:
    path = fetch(f"{split}/anno_{split}.pickle")
    annotations = pickle.loads(path.read_bytes())
    differences, depths, focals, sample_max, eligible = [], [], [], [], []
    for key,a in annotations.items():
        xyz, uv, K = a["xyz"].astype(np.float64), a["uv_vis"][:,:2].astype(np.float64), a["K"].astype(np.float64)
        p = xyz@K.T
        delta = p[:,:2]/p[:,2:]-uv
        differences.append(delta)
        depths.append(xyz[:,2])
        focals.append([K[0,0],K[1,1]])
        sample_max.append([int(key),float(np.linalg.norm(delta,axis=-1).max())])
        right_uv, right_xyz = a["uv_vis"][21:42], xyz[21:42]
        if ((right_uv[:,2]>0).sum() >= settings["min_visible"]
                and ((right_uv[:,:2]>=0)&(right_uv[:,:2]<320)).all()
                and np.ptp(right_uv[:,:2],axis=0).max() >= settings["min_extent_px"]
                and (right_xyz[:,2]>0).all()):
            eligible.append(int(key))
    d = np.concatenate(differences)
    norms = np.linalg.norm(d,axis=1)
    selected = np.random.default_rng(settings["seed"]).choice(sorted(eligible),settings["subset_counts"][split],replace=False)
    selected_delta, selected_z = [], []
    for key in selected:
        a = annotations[int(key)]
        p = a["xyz"][21:42].astype(np.float64) @ a["K"].astype(np.float64).T
        selected_delta.extend(np.linalg.norm(p[:,:2]/p[:,2:]-a["uv_vis"][21:42,:2],axis=-1).tolist())
        selected_z.extend(a["xyz"][21:42,2].tolist())
    report[split] = {
        "images":len(annotations),"annotation_dtypes":{k:str(v.dtype) for k,v in next(iter(annotations.values())).items()},
        "projection_error_quantiles_px":dict(zip(["min","p50","p90","p99","max"],np.quantile(norms,[0,.5,.9,.99,1]).tolist())),
        "mean_signed_delta_uv_px":d.mean(0).tolist(),
        "max_signed_delta_uv_px":np.max(np.abs(d),axis=0).tolist(),
        "worst_samples":sorted(sample_max,key=lambda x:x[1],reverse=True)[:10],
        "joint_z_range_m":[float(np.min(depths)),float(np.max(depths))],
        "focal_range_px":[float(np.min(focals)),float(np.max(focals))],
        "eligible_count":len(eligible),"selected_count":len(selected),
        "selected_projection_quantiles_px":np.quantile(selected_delta,[0,.5,.9,.99,1]).tolist(),
        "selected_joint_z_range_m":[min(selected_z),max(selected_z)]}
(ROOT/"artifacts/rhd_annotation_audit_filtered.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
