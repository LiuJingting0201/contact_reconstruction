"""Measure meeting dual-focal claims on the fixed RHD development crops."""
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path
import numpy as np
for name,value in {"bool":np.bool_,"int":int,"float":float,"complex":complex,
                   "object":object,"unicode":str,"str":str}.items():
    if name not in np.__dict__:
        setattr(np,name,value)
import cv2
import torch
from scipy.optimize import least_squares
from yacs.config import CfgNode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"repos/hamer"))
from hamer.models.hamer import HAMER

out = ROOT/"artifacts/teacher_geometry_control_run02"
out.mkdir(exist_ok=False)
for source in [__file__,ROOT/"docs/dyn_hamr_reproduction_protocol.md"]:
    shutil.copy2(source,out)
torch.manual_seed(42)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
cfg = CfgNode.load_cfg((ROOT/"inputs/model_config.yaml").read_text())
cfg.MANO.DATA_DIR = str(ROOT/"inputs/data")
cfg.MANO.MODEL_PATH = str(ROOT/"inputs/data/mano")
cfg.MANO.MEAN_PARAMS = str(ROOT/"inputs/data/mano_mean_params.npz")
cfg.MODEL.BBOX_SHAPE = [192,256]
cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS",None)
model = HAMER(cfg,init_renderer=False)
checkpoint = torch.load(ROOT/"inputs/hamer.ckpt",map_location="cpu")
incompat = model.load_state_dict(checkpoint["state_dict"],strict=False)
assert not any(not k.startswith(("discriminator.","renderer.","mesh_renderer."))
               for k in incompat.missing_keys+incompat.unexpected_keys)
del checkpoint
model = model.cuda().eval()
manifest = json.loads((ROOT/"inputs/rhd/manifest.json").read_text())["splits"]["evaluation"]
target = np.load(ROOT/"artifacts/rhd_supervised_v3/evaluation_targets.npz")
raw = np.load(ROOT/"inputs/rhd/evaluation_subset.npz")
arrays, rows, tiles = {}, [], []


def project(xyz, translation, K):
    camera = xyz+translation
    assert np.all(camera[:,2] > 0)
    homogeneous = camera@K.T
    return homogeneous[:,:2]/homogeneous[:,2:]


for i,row in enumerate(manifest["rows"]):
    image = cv2.cvtColor(cv2.imread(str(ROOT/"inputs/rhd"/row["image"])),cv2.COLOR_BGR2RGB)
    affine = target["affine"][i]
    inverse = cv2.invertAffineTransform(affine)
    crop = cv2.warpAffine(image,affine,(256,256),flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    normalized = (crop.astype(np.float32)/255-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    batch = {"img":torch.tensor(normalized.transpose(2,0,1)[None],device="cuda")}
    K = raw["K"][i].astype(float)
    assert np.isclose(K[0,0],K[1,1])
    focal_crop = affine[0,0]*K[0,0]
    branches = {}
    with torch.no_grad():
        for name,focal in [("nominal",5000.),("real",float(focal_crop))]:
            cfg.EXTRA.FOCAL_LENGTH = focal
            p = model.forward_step(batch)
            values = {k:v.detach().cpu().numpy() for k,v in p.items() if torch.is_tensor(v)}
            values.update({k:v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
                           for k,v in p["pred_mano_params"][0].items()})
            branches[name] = values
            arrays.update({f"{i}_{name}_{k}":v for k,v in values.items()})
    nominal,real = branches["nominal"],branches["real"]
    for name in ["global_orient","hand_pose","betas","pred_cam","pred_keypoints_3d"]:
        assert np.max(np.abs(nominal[name]-real[name])) < 1e-6
    xyz = nominal["pred_keypoints_3d"][0].astype(float)
    pseudo_crop = (nominal["pred_keypoints_2d"][0]+.5)*256
    pseudo = np.c_[pseudo_crop,np.ones(21)]@inverse.T
    gt = raw["uv_vis"][i,:,:2]
    visible = target["visible"][i]
    center = np.r_[128.,128.,1.]@inverse.T
    translation = real["pred_cam_t"][0].astype(float).copy()
    translation[:2] += (center-K[:2,2])*translation[2]/K.diagonal()[:2]
    lower_z = -xyz[:,2].min()+1e-4
    assert translation[2] > lower_z

    def fit(observed):
        result = least_squares(lambda t:(project(xyz,t,K)[visible]-observed[visible]).reshape(-1),
                               translation,bounds=([-np.inf,-np.inf,lower_z],[np.inf,np.inf,np.inf]),
                               loss="soft_l1",f_scale=3.,max_nfev=300)
        return result

    fitted,oracle = fit(pseudo),fit(gt)
    states = {"initial":translation,"pseudo_fit":fitted.x,"gt_oracle":oracle.x}
    record = {"id":int(row["id"]),"focal_crop":float(focal_crop),
              "teacher_visible_crop_error_px":float(np.linalg.norm(pseudo_crop-(target["uv_normalized"][i]+.5)*256,axis=-1)[visible].mean()),
              "teacher_relative_mpjpe_mm":float(np.linalg.norm((xyz-xyz[:1])-target["xyz_relative_m"][i],axis=-1).mean()*1000),
              "nominal_wrist_depth_error_mm":float(abs(xyz[0,2]+nominal["pred_cam_t"][0,2]-raw["xyz"][i,0,2])*1000),
              "fits":{"pseudo_success":bool(fitted.success),"oracle_success":bool(oracle.success)}}
    for name,t in states.items():
        uv = project(xyz,t,K)
        record[name] = {
            "visible_full_error_px":float(np.linalg.norm(uv-gt,axis=-1)[visible].mean()),
            "visible_crop_error_px":float(np.linalg.norm(uv-gt,axis=-1)[visible].mean()*affine[0,0]),
            "visible_pseudo_residual_full_px":float(np.linalg.norm(uv-pseudo,axis=-1)[visible].mean()),
            "wrist_depth_abs_error_mm":float(abs(xyz[0,2]+t[2]-raw["xyz"][i,0,2])*1000),
            "translation_m":t.tolist()}
        arrays[f"{i}_{name}_uv_full"] = uv
        arrays[f"{i}_{name}_translation_m"] = t
    arrays[f"{i}_pseudo_full"] = pseudo
    rows.append(record)
    if i < 8:
        tile = crop.copy()
        for points,color in [((target["uv_normalized"][i]+.5)*256,(0,255,0)),(pseudo_crop,(255,0,0))]:
            for x,y in points:
                cv2.circle(tile,(round(float(x)),round(float(y))),2,color,-1)
        tiles.append(tile)
    # #region debug-point E:teacher-geometry
    if os.environ.get("DEBUG_SERVER_URL"):
        urllib.request.urlopen(urllib.request.Request(os.environ["DEBUG_SERVER_URL"],
            data=json.dumps({"sessionId":"hand-camera-collapse","runId":out.name,
                "hypothesisId":"E","location":"teacher-dual-focal","msg":"[DEBUG] teacher and real-K geometry",
                "data":record}).encode(),headers={"Content-Type":"application/json"}),timeout=10).read()
    # #endregion
    with (out/"per_image.jsonl").open("a") as f:
        f.write(json.dumps(record)+"\n")

summary = {k:float(np.mean([r[k] for r in rows])) for k in [
    "teacher_visible_crop_error_px","teacher_relative_mpjpe_mm","nominal_wrist_depth_error_mm"]}
for name in states:
    summary[name] = {k:float(np.mean([r[name][k] for r in rows])) for k in rows[0][name] if k != "translation_m"}
report = {"mean_of_per_image_metrics":summary,"rows":rows,
          "checkpoint_missing_keys":incompat.missing_keys,"checkpoint_unexpected_keys":incompat.unexpected_keys,
          "limitations":"Synthetic development subset; GT square boxes; teacher training overlap unknown; translation-only fitting, no stereo. Mean per-image visible error differs from pooled visible-joint metric."}
np.savez_compressed(out/"outputs.npz",**arrays)
(out/"results.json").write_text(json.dumps(report,indent=2)+"\n")
(out/"effective_config.yaml").write_text(cfg.dump())
cv2.imwrite(str(out/"teacher_overlay.jpg"),cv2.cvtColor(np.concatenate(tiles,axis=1),cv2.COLOR_RGB2BGR))
