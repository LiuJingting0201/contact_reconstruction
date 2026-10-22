"""Preregistered small RHD GT-supervised ResNet/MANO pilot; no teacher."""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

for name, value in {"bool": np.bool_, "int": int, "float": float, "complex": complex,
                    "object": object, "unicode": str, "str": str}.items():
    if name not in np.__dict__:
        setattr(np, name, value)
import cv2
import torch
import yaml
from yacs.config import CfgNode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"repos/Fast-HaMeR"))
from hamer.models.hamer import HAMER


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def positive_camera_hook(module, inputs, output):
    params, camera, history = output
    camera = torch.cat([torch.nn.functional.softplus(camera[:,:1])+.05,camera[:,1:]],dim=1)
    history = dict(history)
    if "cam" in history:
        raw = history["cam"]
        history["cam"] = torch.cat([torch.nn.functional.softplus(raw[:,:1])+.05,raw[:,1:]],dim=1)
    return params, camera, history


def configure_camera(model, settings):
    if settings.get("positive_camera", False):
        model.mano_head.register_forward_hook(positive_camera_hook)


def geometry_depth(prediction, enforce):
    depth = prediction["pred_keypoints_3d"][:,:,2]+prediction["pred_cam_t"][:,None,2]
    if enforce:
        assert torch.isfinite(depth).all() and (depth > 0).all(), "invalid camera-space depth"
        assert torch.isfinite(prediction["pred_cam"]).all() and (prediction["pred_cam"][:,0] > 0).all()
        assert torch.isfinite(prediction["pred_keypoints_2d"]).all()
    return depth


def load_data(split, settings, out):
    data_root = ROOT/"inputs/rhd"
    manifest = json.loads((data_root/"manifest.json").read_text())["splits"][split]
    path = data_root/f"{split}_subset.npz"
    assert sha(path) == manifest["labels_sha256"]
    raw = np.load(path)
    images, uv, affine = [], [], []
    n = settings["image_size"]
    preview = []
    for i, row in enumerate(manifest["rows"]):
        assert raw["ids"][i] == row["id"]
        image_path = data_root/row["image"]
        assert sha(image_path) == row["sha256"]
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        points = raw["uv_vis"][i,:,:2]
        lo, hi = points.min(0), points.max(0)
        center = (lo+hi)/2
        side = (hi-lo).max()*settings["crop_rescale"]
        scale = n/side
        transform = np.array([[scale,0,n/2-scale*center[0]],
                              [0,scale,n/2-scale*center[1]]],np.float32)
        crop = cv2.warpAffine(image,transform,(n,n),flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        target = np.c_[points,np.ones(len(points))] @ transform.T
        # The inverse must recover the source UV independently of model execution.
        recovered = np.c_[target,np.ones(len(points))] @ cv2.invertAffineTransform(transform).T
        assert np.max(np.abs(recovered-points)) < 1e-3
        if i < 8:
            overlay = crop.copy()
            for j, (x,y) in enumerate(target):
                color = (0,255,0) if raw["uv_vis"][i,j,2] else (255,128,0)
                cv2.circle(overlay,(round(x),round(y)),2,color,-1)
                cv2.putText(overlay,str(j),(round(x)+2,round(y)),cv2.FONT_HERSHEY_SIMPLEX,.25,color,1)
            preview.append(overlay)
        normalized = (crop.astype(np.float32)/255-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
        images.append(normalized.transpose(2,0,1))
        uv.append(target/n-.5)
        affine.append(transform)
    cv2.imwrite(str(out/f"{split}_gt_preview.jpg"),cv2.cvtColor(np.concatenate(preview,axis=1),cv2.COLOR_RGB2BGR))
    xyz = raw["xyz"].astype(np.float32)
    saved = {"ids":raw["ids"],"xyz_absolute_m":xyz,"xyz_relative_m":xyz-xyz[:,:1],
             "uv_normalized":np.array(uv,dtype=np.float32),"visible":raw["uv_vis"][:,:,2]>0,
             "affine":np.array(affine),"K":raw["K"]}
    np.savez_compressed(out/f"{split}_targets.npz",**saved)
    # #region debug-point B:data-conventions
    if os.environ.get("DEBUG_SERVER_URL"):
        import urllib.request
        urllib.request.urlopen(urllib.request.Request(os.environ["DEBUG_SERVER_URL"], data=json.dumps({"sessionId":"hand-camera-collapse","runId":out.name,"hypothesisId":"B","location":"load_data","msg":"[DEBUG] crop and coordinate evidence","data":{"split":split,"ids":saved["ids"].tolist(),"xyz_range_m":[float(xyz.min()),float(xyz.max())],"z_range_m":[float(xyz[:,:,2].min()),float(xyz[:,:,2].max())],"crop_span_px":np.ptp(saved["uv_normalized"],axis=1).max(1).tolist(),"first_K":saved["K"][0].tolist(),"first_affine":saved["affine"][0].tolist()}}).encode(),headers={"Content-Type":"application/json"}),timeout=10).read()
    # #endregion
    return {"img":torch.tensor(np.array(images),device="cuda"),
            "xyz":torch.tensor(saved["xyz_relative_m"],device="cuda"),
            "uv":torch.tensor(saved["uv_normalized"],device="cuda"),
            "visible":torch.tensor(saved["visible"],device="cuda"),
            "ids":raw["ids"]}


def evaluate(model, datasets, step, out, batch_size):
    model.eval()
    report = {"step":step,"splits":{}}
    with torch.no_grad():
        for split, data in datasets.items():
            xyzs, uvs, cams, depths = [], [], [], []
            for begin in range(0,len(data["ids"]),batch_size):
                p = model.forward_step({"img":data["img"][begin:begin+batch_size]})
                xyz = p["pred_keypoints_3d"]
                xyzs.append((xyz-xyz[:,:1]).cpu().numpy())
                uvs.append(p["pred_keypoints_2d"].cpu().numpy())
                cams.append(p["pred_cam"].cpu().numpy())
                depths.append(geometry_depth(p,model.cfg.MODEL.get("POSITIVE_CAMERA",False)).cpu().numpy())
            xyz, uv = np.concatenate(xyzs), np.concatenate(uvs)
            camera = np.concatenate(cams)
            assert np.isfinite(xyz).all() and np.isfinite(uv).all()
            target_xyz, target_uv = data["xyz"].cpu().numpy(), data["uv"].cpu().numpy()
            visible = data["visible"].cpu().numpy()
            err3d = np.linalg.norm(xyz-target_xyz,axis=-1)*1000
            err2d = np.linalg.norm(uv-target_uv,axis=-1)*256
            np.savez_compressed(out/f"{split}_predictions_{step:04d}.npz",
                                ids=data["ids"],xyz_relative_m=xyz,uv_normalized=uv,
                                error3d_mm=err3d,error2d_px=err2d,visible=visible,
                                weak_camera=camera,camera_space_joint_depth_m=np.concatenate(depths))
            report["splits"][split] = {
                "mpjpe_mm":float(err3d.mean()),
                "visible_crop_error_px":float(err2d[visible].mean()),
                "predicted_bbox_span_px_min_median_max":np.quantile(np.ptp(uv,axis=1).max(1)*256,[0,.5,1]).tolist(),
                "weak_scale_min_median_max":np.quantile(camera[:,0],[0,.5,1]).tolist(),
                "camera_space_depth_min_max_m":[float(np.min(depths)),float(np.max(depths))],
                "per_image_mpjpe_mm":err3d.mean(1).tolist(),
                "per_image_visible_crop_error_px":((err2d*visible).sum(1)/visible.sum(1)).tolist()}
            # #region debug-point A:camera-evaluation
            if os.environ.get("DEBUG_SERVER_URL"):
                import urllib.request
                urllib.request.urlopen(urllib.request.Request(os.environ["DEBUG_SERVER_URL"], data=json.dumps({"sessionId":"hand-camera-collapse","runId":out.name,"hypothesisId":"A","location":"evaluate","msg":"[DEBUG] camera and projection at evaluation","data":{"step":step,"split":split,"camera":camera.tolist(),"metrics":report["splits"][split]}}).encode(),headers={"Content-Type":"application/json"}),timeout=10).read()
            # #endregion
            if split == "evaluation":
                tiles = []
                for i in range(8):
                    rgb = data["img"][i].cpu().numpy().transpose(1,2,0)
                    rgb = np.ascontiguousarray(np.clip((rgb*np.array([.229,.224,.225])+np.array([.485,.456,.406]))*255,0,255).astype(np.uint8))
                    for points, color in [(target_uv[i],(0,255,0)),(uv[i],(255,0,0))]:
                        for x,y in (points+.5)*256:
                            cv2.circle(rgb,(round(float(x)),round(float(y))),2,color,-1)
                    tiles.append(rgb)
                cv2.imwrite(str(out/f"evaluation_overlay_{step:04d}.jpg"),
                            cv2.cvtColor(np.concatenate(tiles,axis=1),cv2.COLOR_RGB2BGR))
    with (out/"evaluations.jsonl").open("a") as f:
        f.write(json.dumps(report)+"\n")
    print(json.dumps(report),flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/rhd_supervised_v4.json")
    parser.add_argument("--run-id")
    parser.add_argument("--deterministic",action=argparse.BooleanOptionalAction,default=True)
    args = parser.parse_args()
    config_path = ROOT/args.config
    settings = json.loads(config_path.read_text())
    settings["run_id"] = args.run_id or settings["run_id"]
    settings["deterministic"] = args.deterministic
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
    out = ROOT/"artifacts"/settings["run_id"]
    out.mkdir(exist_ok=False)
    for source in [Path(__file__),config_path,ROOT/"docs/dyn_hamr_reproduction_protocol.md",ROOT/"inputs/rhd/manifest.json"]:
        shutil.copy2(source,out/source.name)
    torch.manual_seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = CfgNode.load_cfg((ROOT/"inputs/model_config.yaml").read_text())
    override = yaml.safe_load((ROOT/"repos/Fast-HaMeR/hamer/configs_hydra/experiment/resnet_50.yaml").read_text())
    cfg.MODEL.BACKBONE = CfgNode(override["MODEL"]["BACKBONE"])
    cfg.MODEL.MANO_HEAD = CfgNode(override["MODEL"]["MANO_HEAD"])
    cfg.MODEL.IMAGE_SIZE = settings["image_size"]
    cfg.MODEL.POSITIVE_CAMERA = settings.get("positive_camera",False)
    cfg.MANO.DATA_DIR = str(ROOT/"inputs/data")
    cfg.MANO.MODEL_PATH = str(ROOT/"inputs/data/mano")
    cfg.MANO.MEAN_PARAMS = str(ROOT/"inputs/data/mano_mean_params.npz")
    cfg.EXTRA.FOCAL_LENGTH = 5000.
    cfg.LOSS_WEIGHTS.ADVERSARIAL = 0.
    cfg.LOSS_WEIGHTS.KEYPOINTS_3D = settings["weights"]["joints3d"]
    cfg.LOSS_WEIGHTS.KEYPOINTS_2D = settings["weights"]["joints2d"]
    cfg.LOSS_WEIGHTS.GLOBAL_ORIENT = 0.
    cfg.LOSS_WEIGHTS.HAND_POSE = 0.
    cfg.LOSS_WEIGHTS.BETAS = settings["weights"]["shape_prior"]
    cfg.TRAIN.LR = settings["lr"]
    cfg.TRAIN.WEIGHT_DECAY = settings["weight_decay"]
    cfg.TRAIN.BATCH_SIZE = settings["batch_size"]
    torch.hub.set_dir(str(ROOT/"inputs/torch_hub"))
    weight = ROOT/"inputs/resnet50-11ad3fa6.pth"
    assert sha(weight).startswith("11ad3fa6")
    cache = ROOT/"inputs/torch_hub/checkpoints"/weight.name
    cache.parent.mkdir(parents=True,exist_ok=True)
    if not cache.exists():
        os.link(weight,cache)
    model = HAMER(cfg,init_renderer=False).cuda()
    if "head_init" in settings:
        init = settings["head_init"]
        for name in ["decpose","decshape","deccam"]:
            layer = getattr(model.mano_head,name)
            torch.nn.init.xavier_uniform_(layer.weight,gain=init["xavier_gain"])
            if init["zero_readout_bias"]:
                torch.nn.init.zeros_(layer.bias)
        with torch.no_grad():
            model.mano_head.init_cam.copy_(torch.tensor([init["weak_camera"]],device="cuda"))
            if settings.get("positive_camera", False):
                scale = model.mano_head.init_cam[:,0]-.05
                assert (scale > 0).all()
                model.mano_head.init_cam[:,0].copy_(scale+torch.log(-torch.expm1(-scale)))
    configure_camera(model, settings)
    parameters = model.get_parameters()
    # #region debug-point A:parameter-initialization
    if os.environ.get("DEBUG_SERVER_URL"):
        import urllib.request
        urllib.request.urlopen(urllib.request.Request(os.environ["DEBUG_SERVER_URL"], data=json.dumps({"sessionId":"hand-camera-collapse","runId":out.name,"hypothesisId":"A","location":"main/model-init","msg":"[DEBUG] actual initialization and parameter ownership","data":{"first_optimizer_parameter":[n for n,p in model.named_parameters() if p is parameters[0]],"mean_camera":model.mano_head.init_cam.detach().cpu().tolist(),"mean_pose":model.mano_head.init_hand_pose.detach().cpu().tolist(),"settings":settings}}).encode(),headers={"Content-Type":"application/json"}),timeout=10).read()
    # #endregion
    first = parameters[0].detach().clone()
    backbone_name, backbone_parameter = next(model.backbone.named_parameters())
    backbone_first = backbone_parameter.detach().clone()
    optimizer = torch.optim.AdamW(parameters,lr=settings["lr"],weight_decay=settings["weight_decay"])
    datasets = {s:load_data(s,settings,out) for s in settings["subset_counts"]}
    (out/"effective_config.yaml").write_text(cfg.dump())
    start = time.perf_counter()
    start_utc = datetime.now(timezone.utc).isoformat()
    initial = evaluate(model,datasets,0,out,settings["batch_size"])
    data = datasets["training"]
    rng = np.random.default_rng(settings["seed"])
    rows, permutation, cursor = [], None, len(data["ids"])
    for step in range(1,settings["steps"]+1):
        if cursor+settings["batch_size"] > len(data["ids"]):
            permutation = rng.permutation(len(data["ids"]))
            cursor = 0
        indices = permutation[cursor:cursor+settings["batch_size"]]
        cursor += settings["batch_size"]
        model.train()
        for module in model.modules():
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        optimizer.zero_grad(set_to_none=True)
        prediction = model.forward_step({"img":data["img"][indices]},train=True)
        geometry_depth(prediction,settings.get("positive_camera",False))
        xyz = prediction["pred_keypoints_3d"]
        visible = data["visible"][indices]
        terms = {
            "joints3d":((xyz-xyz[:,:1])-data["xyz"][indices]).square().mean(),
            "joints2d":(prediction["pred_keypoints_2d"]-data["uv"][indices]).square()[visible].mean(),
            "shape_prior":prediction["pred_mano_params"]["betas"].square().mean()}
        loss = sum(v*settings["weights"][k] for k,v in terms.items())
        # #region debug-point C:component-gradients
        if os.environ.get("DEBUG_SERVER_URL") and step in (1,2,10,50,100,200):
            import urllib.request
            _dp = {"pose_readout":model.mano_head.decpose.weight,"camera_readout":model.mano_head.deccam.weight,"backbone_first":next(model.backbone.parameters())}
            _dg = {k:torch.autograd.grad(v*settings["weights"][k],tuple(_dp.values()),retain_graph=True,allow_unused=True) for k,v in terms.items()}
            _norms = {k:{n:None if g is None else float(g.detach().norm()) for n,g in zip(_dp,gs)} for k,gs in _dg.items()}
            _cosines = {n:None if a is None or b is None else float(torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),dim=0)) for n,a,b in zip(_dp,_dg["joints3d"],_dg["joints2d"])}
            urllib.request.urlopen(urllib.request.Request(os.environ["DEBUG_SERVER_URL"], data=json.dumps({"sessionId":"hand-camera-collapse","runId":out.name,"hypothesisId":"C","location":"main/pre-backward","msg":"[DEBUG] weighted component gradients without optimizer changes","data":{"step":step,"sample_ids":data["ids"][indices].tolist(),"gradient_norms":_norms,"cosine_3d_2d":_cosines,"weak_camera":prediction["pred_cam"].detach().cpu().tolist(),"crop_camera_translation":prediction["pred_cam_t"].detach().cpu().tolist(),"terms":{k:float(v.detach()) for k,v in terms.items()}}}).encode(),headers={"Content-Type":"application/json"}),timeout=10).read()
        # #endregion
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters,settings["clip_grad_norm"],error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        row = {"step":step,"sample_ids":data["ids"][indices].tolist(),"loss":float(loss.detach()),
               "terms":{k:float(v.detach()) for k,v in terms.items()},
               "grad_norm_before_clip":float(norm),"elapsed_s":time.perf_counter()-start}
        rows.append(row)
        with (out/"steps.jsonl").open("a") as f:
            f.write(json.dumps(row)+"\n")
        print(json.dumps(row),flush=True)
        if step % settings["eval_interval"] == 0:
            final = evaluate(model,datasets,step,out,settings["batch_size"])
    update = float((parameters[0].detach()-first).abs().max())
    backbone_update = float((backbone_parameter.detach()-backbone_first).abs().max())
    assert update > 0 and backbone_update > 0
    model.eval()
    with torch.no_grad():
        before_output = model.forward_step({"img":datasets["evaluation"]["img"][:1]})
        before = before_output["pred_keypoints_3d"].cpu()
        before_uv = before_output["pred_keypoints_2d"].cpu()
    checkpoint = out/"last.pt"
    torch.save({"state_dict":model.state_dict(),"optimizer":optimizer.state_dict(),
                "steps":len(rows),"settings":settings,"cfg":cfg.dump(),
                "torch_rng_state":torch.get_rng_state(),"cuda_rng_state":torch.cuda.get_rng_state(),
                "numpy_generator_state":rng.bit_generator.state,
                "permutation":permutation,"cursor":cursor},checkpoint)
    del model, optimizer, parameters
    torch.cuda.empty_cache()
    restored = HAMER(cfg,init_renderer=False).cuda().eval()
    configure_camera(restored,settings)
    restored.load_state_dict(torch.load(checkpoint,map_location="cpu")["state_dict"],strict=True)
    with torch.no_grad():
        after_output = restored.forward_step({"img":datasets["evaluation"]["img"][:1]})
        after = after_output["pred_keypoints_3d"].cpu()
        after_uv = after_output["pred_keypoints_2d"].cpu()
    difference = float((after-before).abs().max())
    uv_difference = float((after_uv-before_uv).abs().max())
    assert difference < 1e-6 and uv_difference < 1e-6
    report = {"kind":"RHD_GT_supervised_subset_pilot_no_KD","settings":settings,
              "start_utc":start_utc,"end_utc":datetime.now(timezone.utc).isoformat(),
              "initial":initial,"final":final,"actual_updates":len(rows),
              "max_first_parameter_update":update,"reload_joint_max_diff_m":difference,
              "backbone_parameter_name":backbone_name,"max_backbone_parameter_update":backbone_update,
              "reload_uv_max_diff_normalized":uv_difference,
              "wall_s_including_evaluation_save_reload":time.perf_counter()-start,
              "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
              "source_sha256":{str(p.relative_to(ROOT)):sha(p) for p in [
                  Path(__file__),config_path,ROOT/"inputs/rhd/manifest.json",weight,
                  ROOT/"inputs/data/mano/MANO_RIGHT.pkl"]},
              "limitations":settings["scope"]}
    (out/"results.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


if __name__ == "__main__":
    main()
