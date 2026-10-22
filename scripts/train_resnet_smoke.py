"""20-step ResNet/MANO training on one real crop with frozen teacher labels.

An overfit plumbing test, not the no-KD supervised baseline or an accuracy eval.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
for name, value in {"bool": np.bool_, "int": int, "float": float, "complex": complex,
                    "object": object, "unicode": str, "str": str}.items():
    if name not in np.__dict__:
        setattr(np, name, value)
import torch
import yaml
from yacs.config import CfgNode

root = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/"repos/Fast-HaMeR"))
from hamer.models.hamer import HAMER

settings = json.loads((root/"experiments/real_image_smoke.json").read_text())
train = settings["train_smoke"]
out = root/"artifacts/resnet_pseudolabel_smoke_v1"
out.mkdir(exist_ok=False)
torch.manual_seed(settings["seed"])
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
cfg = CfgNode.load_cfg((root/"inputs/model_config.yaml").read_text())
override = yaml.safe_load((root/"repos/Fast-HaMeR/hamer/configs_hydra/experiment/resnet_50.yaml").read_text())
override.pop("defaults")
cfg.MODEL.BACKBONE = CfgNode(override["MODEL"]["BACKBONE"])
cfg.MODEL.MANO_HEAD = CfgNode(override["MODEL"]["MANO_HEAD"])
cfg.MANO.DATA_DIR = str(root/"inputs/data")
cfg.MANO.MODEL_PATH = str(root/"inputs/data/mano")
cfg.MANO.MEAN_PARAMS = str(root/"inputs/data/mano_mean_params.npz")
cfg.EXTRA.FOCAL_LENGTH = 5000.
cfg.LOSS_WEIGHTS.ADVERSARIAL = 0.
cfg.TRAIN.LR = train["lr"]
cfg.TRAIN.WEIGHT_DECAY = train["weight_decay"]
cfg.TRAIN.BATCH_SIZE = train["batch_size"]
torch.hub.set_dir(str(root/"inputs/torch_hub"))
weight_cache = root/"inputs/torch_hub/checkpoints"
weight_cache.mkdir(parents=True,exist_ok=True)
weight_path = root/"inputs/resnet50-11ad3fa6.pth"
weight_hash = hashlib.sha256(weight_path.read_bytes()).hexdigest()
assert weight_hash.startswith("11ad3fa6"), "ImageNet pretrained weight hash mismatch"
cached = weight_cache/weight_path.name
if not cached.exists():
    os.link(weight_path,cached)
target_path = root/"artifacts/real_image_smoke_v1/frozen_pseudo_targets.pt"
targets = torch.load(target_path,map_location="cpu")
image = targets["image"].repeat(train["batch_size"],1,1,1).cuda()
target_j3d = targets["5000_pred_keypoints_3d"].cuda()
target_j2d = targets["5000_pred_keypoints_2d"].cuda()
target_pose = {k:targets["5000_"+k].cuda() for k in ["global_orient","hand_pose","betas"]}
model = HAMER(cfg, init_renderer=False).cuda()
params = model.get_parameters()
optimizer = torch.optim.AdamW(params,lr=train["lr"],weight_decay=train["weight_decay"])
first_parameter = params[0].detach().clone()
rows = []
model.train()
start = time.perf_counter()
for step in range(train["steps"]):
    optimizer.zero_grad(set_to_none=True)
    prediction = model.forward_step({"img":image},train=True)
    p = prediction["pred_mano_params"]
    terms = {
        "joints3d": (prediction["pred_keypoints_3d"]-target_j3d).square().mean(),
        "joints2d": (prediction["pred_keypoints_2d"]-target_j2d).square().mean(),
        "pose": sum((p[k]-target_pose[k]).square().mean() for k in ["global_orient","hand_pose"]),
        "shape": (p["betas"]-target_pose["betas"]).square().mean(),
    }
    loss = sum(terms[k]*train["weights"][k] for k in terms)
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite_training_loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(params,train["clip_grad_norm"],error_if_nonfinite=True)
    optimizer.step()
    torch.cuda.synchronize()
    row = {"step":step+1,"loss":loss.item(),"grad_norm_before_clip":float(norm),
           "terms":{k:float(v.detach()) for k,v in terms.items()},
           "elapsed_s":time.perf_counter()-start}
    rows.append(row)
    with (out/"steps.jsonl").open("a") as f:
        f.write(json.dumps(row)+"\n")
    print(json.dumps(row),flush=True)
update = float((params[0].detach()-first_parameter).abs().max())
assert update > 0
model.eval()
with torch.no_grad():
    before_reload = model.forward_step({"img":image[:1]})["pred_keypoints_3d"].cpu()
checkpoint = out/"last.pt"
torch.save({"state_dict":model.state_dict(),"optimizer":optimizer.state_dict(),
            "steps":train["steps"],"cfg":cfg.dump(),"settings":settings},checkpoint)
del optimizer, model, params
torch.cuda.empty_cache()
reloaded = HAMER(cfg,init_renderer=False).cuda().eval()
reloaded.load_state_dict(torch.load(checkpoint,map_location="cpu")["state_dict"],strict=True)
with torch.no_grad():
    after_reload = reloaded.forward_step({"img":image[:1]})["pred_keypoints_3d"].cpu()
diff = float((after_reload-before_reload).abs().max())
assert diff < 1e-6
report = {"kind":"one_real_crop_offline_teacher_pseudolabel_training_smoke",
          "not_a_noKD_GT_baseline":True,"unique_images":1,"steps":len(rows),
          "settings":settings,"initial_train_loss":rows[0]["loss"],"last_train_loss":rows[-1]["loss"],
          "max_first_parameter_update":update,"reload_joint_max_diff_m":diff,
          "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
          "wall_s_including_checkpoint_reload":time.perf_counter()-start,
          "source_hashes":{"teacher_targets":hashlib.sha256(target_path.read_bytes()).hexdigest(),
                           "imagenet_weights":weight_hash,"script":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
          "limitations":"20 AdamW steps on repeated same crop; no independent validation, no public benchmark, no distillation comparison"}
(out/"results.json").write_text(json.dumps(report,indent=2)+"\n")
(out/"effective_config.yaml").write_text(cfg.dump())
print(json.dumps(report,indent=2))
