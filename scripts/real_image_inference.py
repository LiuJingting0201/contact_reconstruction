"""Real HaMeR forward on an author-provided sample; no GT accuracy claims."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

# chumpy in the author's MANO pickle references NumPy aliases removed in 1.24.
# Process-local compatibility only; no installed package is modified.
for name, value in {"bool": np.bool_, "int": int, "float": float, "complex": complex,
                    "object": object, "unicode": str, "str": str}.items():
    if name not in np.__dict__:
        setattr(np, name, value)
import cv2
import torch
from yacs.config import CfgNode

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/"repos/hamer"))
from hamer.models.hamer import HAMER
from hamer.datasets.vitdet_dataset import ViTDetDataset

settings = json.loads((root/"experiments/real_image_smoke.json").read_text())
out = root/"artifacts/real_image_smoke_v1"
out.mkdir(exist_ok=False)
torch.manual_seed(settings["seed"])
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
cfg = CfgNode.load_cfg((root/"inputs/model_config.yaml").read_text())
cfg.MANO.DATA_DIR = str(root/"inputs/data")
cfg.MANO.MODEL_PATH = str(root/"inputs/data/mano")
cfg.MANO.MEAN_PARAMS = str(root/"inputs/data/mano_mean_params.npz")
cfg.MODEL.BBOX_SHAPE = [192, 256]
cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS", None)
cfg.EXTRA.FOCAL_LENGTH = 500.
checkpoint = root/"inputs/hamer.ckpt"
torch.set_num_threads(4)
model = HAMER(cfg, init_renderer=False)
loaded = torch.load(checkpoint, map_location="cpu")
incompat = model.load_state_dict(loaded["state_dict"], strict=False)
del loaded
if any(not k.startswith(("discriminator.", "renderer.", "mesh_renderer.")) for k in incompat.missing_keys+incompat.unexpected_keys):
    raise RuntimeError(f"Unexplained checkpoint keys: {incompat}")
model = model.to(settings["device"]).eval()
image = cv2.imread(str(root/settings["source_image"]))
dataset = ViTDetDataset(cfg, image, np.array([settings["box_xyxy"]], np.float32),
                       np.array([1]), np.zeros((1, 21, 3), np.float32),
                       rescale_factor=settings["crop_rescale"])
item = dataset[0]
cv2.imwrite(str(out/"crop.png"), np.clip(item["img_patch"][:, :, ::-1],0,255).astype(np.uint8))
batch = {"img": torch.as_tensor(item["img"][None], device=settings["device"])}
np.savez(out/"crop_input.npz", image_tensor=item["img"], inv_trans=item["inv_trans"],
         box_center=item["box_center"], box_size=item["box_size"])
results, arrays, parameters = [], {}, {}
with torch.no_grad():
    for focal in settings["focal_cfg_values"]:
        cfg.EXTRA.FOCAL_LENGTH = focal
        torch.cuda.synchronize()
        start = time.perf_counter()
        prediction = model.forward_step(batch)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        label = str(int(focal))
        tensors = {k:v.detach().cpu().numpy() for k,v in prediction.items() if torch.is_tensor(v)}
        parameters[label] = prediction["pred_mano_params"][0]
        arrays.update({label+"_"+k:v for k,v in tensors.items()})
        arrays.update({label+"_"+k:v for k,v in parameters[label].items()})
        joints_crop = (tensors["pred_keypoints_2d"][0]+.5)*256.
        uv = np.c_[joints_crop, np.ones(21)] @ item["inv_trans"].T
        arrays[label+"_uv_full"] = uv
        canvas = image.copy()
        for x, y in uv:
            cv2.circle(canvas,(int(round(x)),int(round(y))),3,(0,100,255),-1)
        cv2.imwrite(str(out/(label+"_overlay.jpg")),canvas)
        results.append({"focal_cfg":focal, "forward_wall_s":elapsed,
                        "pred_cam":tensors["pred_cam"].tolist(),
                        "pred_cam_t_crop":tensors["pred_cam_t"].tolist(),
                        "finite":all(np.isfinite(v).all().item() for v in tensors.values())})
    assert all(r["finite"] for r in results)
    max_pose_diff = max(float(np.max(abs(parameters["500"][k]-parameters["5000"][k]))) for k in parameters["500"])
    cam_diff = float(np.max(abs(arrays["500_pred_cam"]-arrays["5000_pred_cam"])))
    assert max_pose_diff < 1e-6 and cam_diff < 1e-6
    target = {k: torch.from_numpy(v) for k,v in arrays.items() if k.startswith("5000_")}
    target["image"] = batch["img"].cpu()
    torch.save(target,out/"frozen_pseudo_targets.pt")
np.savez_compressed(out/"outputs.npz",**arrays)
report = {"data_kind":"real_image_no_GT", "settings":settings,
          "checkpoint_missing_keys":incompat.missing_keys,"checkpoint_unexpected_keys":incompat.unexpected_keys,
          "observations":results,"max_MANO_parameter_difference":max_pose_diff,
          "max_weak_camera_difference":cam_diff,
          "mean_full_image_projection_difference_px":float(np.linalg.norm(arrays["500_uv_full"]-arrays["5000_uv_full"],axis=1).mean()),
          "peak_gpu_allocated_bytes":torch.cuda.max_memory_allocated(),
          "limitations":"manual box, single sample, uncalibrated image; timings include cold-start and CPU serialization",
          "source_hashes":{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
              root/settings["source_image"], root/"inputs/data/mano/MANO_RIGHT.pkl", Path(__file__)]}}
(out/"results.json").write_text(json.dumps(report,indent=2)+"\n")
(out/"effective_config.yaml").write_text(cfg.dump())
print(json.dumps(report,indent=2))
