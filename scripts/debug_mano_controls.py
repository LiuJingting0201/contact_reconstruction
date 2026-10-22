"""Direct MANO controls: local synthetic self-fit and RHD GT-oracle fitting."""
import json
import os
import shutil
import urllib.request
import numpy as np
from train_rhd_supervised import ROOT, torch, CfgNode
from hamer.models.mano_wrapper import MANO
from hamer.utils.geometry import rot6d_to_rotmat

out = ROOT/"artifacts/mano_debug_controls"
out.mkdir(exist_ok=False)
for source in [__file__, ROOT/"docs/dyn_hamr_reproduction_protocol.md"]:
    shutil.copy2(source, out)
torch.manual_seed(42)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
cfg = CfgNode.load_cfg((ROOT/"inputs/model_config.yaml").read_text())
cfg.MANO.DATA_DIR = str(ROOT/"inputs/data")
cfg.MANO.MODEL_PATH = str(ROOT/"inputs/data/mano")
cfg.MANO.MEAN_PARAMS = str(ROOT/"inputs/data/mano_mean_params.npz")
mano = MANO(**{k.lower():v for k,v in dict(cfg.MANO).items()}).cuda()
mean = torch.tensor(np.load(cfg.MANO.MEAN_PARAMS)["pose"], device="cuda").float().reshape(1, -1)


def joints(pose, betas):
    rotations = rot6d_to_rotmat(pose).reshape(-1,16,3,3)
    xyz = mano(global_orient=rotations[:,:1],hand_pose=rotations[:,1:],
               betas=betas,pose2rot=False).joints
    return xyz-xyz[:,:1], rotations


def fit(label, target, initial_pose, initial_betas, steps):
    pose = initial_pose.clone().requires_grad_()
    betas = initial_betas.clone().requires_grad_()
    optim = torch.optim.Adam([pose, betas],lr=.02)
    rows = []
    for step in range(steps+1):
        xyz, rotations = joints(pose,betas)
        loss = (xyz-target).square().mean()
        error = torch.linalg.vector_norm(xyz-target,dim=-1).mean(1)*1000
        rows.append({"step":step,"mse":float(loss.detach()),
                     "per_image_mpjpe_mm":error.detach().cpu().tolist()})
        if step == steps:
            break
        optim.zero_grad()
        loss.backward()
        assert torch.isfinite(pose.grad).all() and torch.isfinite(betas.grad).all()
        optim.step()
    with (out/f"{label}_steps.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row)+"\n")
    np.savez_compressed(out/f"{label}.npz",target=target.cpu().numpy(),
                        predicted=xyz.detach().cpu().numpy(),pose6d=pose.detach().cpu().numpy(),
                        betas=betas.detach().cpu().numpy(),rotations=rotations.detach().cpu().numpy())
    result = {"initial":rows[0],"final":rows[-1]}
    # #region debug-point B:direct-fit
    if os.environ.get("DEBUG_SERVER_URL"):
        urllib.request.urlopen(urllib.request.Request(os.environ["DEBUG_SERVER_URL"],
            data=json.dumps({"sessionId":"hand-camera-collapse","runId":out.name,
                "hypothesisId":"B","location":"direct-MANO-fit","msg":"[DEBUG] "+label,
                "data":result}).encode(),headers={"Content-Type":"application/json"}),timeout=10).read()
    # #endregion
    return result


truth_pose = mean.expand(4,-1)+torch.randn(4,96,device="cuda")*.25
truth_beta = torch.randn(4,10,device="cuda")*.3
with torch.no_grad():
    target, _ = joints(truth_pose,truth_beta)
synthetic = fit("synthetic",target,truth_pose+torch.randn_like(truth_pose)*.1,
                truth_beta+torch.randn_like(truth_beta)*.1,600)
assert max(synthetic["final"]["per_image_mpjpe_mm"]) < 1.
raw = np.load(ROOT/"inputs/rhd/training_subset.npz")
target = torch.tensor(raw["xyz"][:8],device="cuda").float()
target = target-target[:,:1]
rhd = fit("rhd_oracle",target,mean.expand(8,-1),torch.zeros(8,10,device="cuda"),1000)
report = {"synthetic":synthetic,"rhd_oracle":rhd,"rhd_ids":raw["ids"][:8].tolist(),
          "limitations":"Local self-fit; RHD uses training GT oracle and unconstrained betas, not network accuracy."}
(out/"results.json").write_text(json.dumps(report,indent=2)+"\n")
