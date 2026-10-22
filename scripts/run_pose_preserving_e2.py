"""Cached real-E2 ablation. Never instantiates or runs the HaMeR teacher."""
import argparse
import json
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
from run_real_e2 import torch_setup, solve as original_solve, project_torch, reprojection_torch, seal, write_json
from real_e2_data import require, sha256

ROOT = Path(__file__).resolve().parents[1]


def angle_torch(a, b):
    """SO(3) geodesic radians, stable at identity; no acos clamp dead zone."""
    import torch
    q = a @ b.transpose(-1, -2)
    skew = torch.stack((q[..., 2, 1]-q[..., 1, 2], q[..., 0, 2]-q[..., 2, 0],
                        q[..., 1, 0]-q[..., 0, 1]), -1)
    sine = .5 * torch.linalg.vector_norm(skew, dim=-1)
    cosine = ((q.diagonal(dim1=-2, dim2=-1).sum(-1)-1)*.5).clamp(-1, 1)
    return torch.atan2(sine, cosine)


def solve_anchor(mano, initial, K, T, pseudo, variant, config, out):
    torch, _, tensor, array, geometry, decode, _, _ = torch_setup()
    r0, b0, t0 = map(tensor, initial)
    trans = torch.nn.Parameter(t0.clone())
    six = r0[..., :2].transpose(-1, -2).reshape(1, 16, 6)
    require(torch.allclose(decode(six).reshape_as(r0), r0, atol=1e-6), "6D roundtrip")
    glob, hand = [torch.nn.Parameter(x.clone()) for x in (six[:, :1], six[:, 1:])]
    mode = variant["variables"]
    groups = [{"params": [trans], "lr": config["translation_lr"]},
              {"params": [glob if mode == "orientation" else hand], "lr": config["rotation_shape_lr"]}]
    opt = torch.optim.Adam(groups)
    K, T, pseudo = map(tensor, (K, T, pseudo))
    confidence, weights = tensor(np.ones((2, 21))), tensor(np.ones(2))

    def forward():
        g = decode(glob).reshape(1, 1, 3, 3) if mode == "orientation" else r0[:, :1]
        h = decode(hand).reshape(1, 15, 3, 3) if mode == "fingers" else r0[:, 1:]
        r = torch.cat((g, h), 1)
        xyz = geometry(mano, r, b0)[0] + trans[0]
        uv, cam = project_torch(xyz, K, T)
        reproj = reprojection_torch(uv, pseudo, confidence, weights, True, config["reprojection_scale_px"])
        prior = (r-r0).square().mean()
        anchor = angle_torch(g, r0[:, :1]).square().mean()
        depth = (config["minimum_depth_m"]-cam[..., 2]).clamp_min(0).square().mean()
        loss = reproj + config["pose_prior_weight"]*prior + variant["lambda_rot"]*anchor
        loss = loss + config["depth_prior_weight"]*depth
        return r, xyz, uv, cam, loss, reproj, prior, anchor, depth

    # Verify the new SO(3) term has a finite identity gradient before optimization.
    probe = torch.eye(3, device="cuda", requires_grad=True)
    angle_torch(probe, torch.eye(3, device="cuda")).square().backward()
    require(torch.isfinite(probe.grad).all(), "geodesic identity gradient")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    trace, failure, updates = [], None, 0
    for step in range(config["steps"]+1):
        r, xyz, uv, cam, loss, reproj, pp, anchor, depth = forward()
        if step == 0:
            initial_loss = float(loss.detach())
        if not torch.isfinite(loss):
            failure = f"nonfinite_objective_{step}"
            break
        if step % config["trace_interval"] == 0:
            trace.append({"step": step, "objective": float(loss.detach()), "reprojection": float(reproj.detach()),
                          "legacy_pose_prior": float(pp.detach()), "rotation_anchor_rad2": float(anchor.detach()),
                          "weighted_rotation_anchor": float((variant["lambda_rot"]*anchor).detach()),
                          "depth_penalty": float(depth.detach()), "minimum_depth_m": float(cam[..., 2].min().detach())})
        if step == config["steps"]:
            break
        opt.zero_grad()
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for g in groups for p in g["params"]):
            failure = f"nonfinite_gradient_{step}"
            break
        if step == config["decay_step"]:
            for g in opt.param_groups:
                g["lr"] *= config["decay_factor"]
        opt.step()
        updates += 1
    torch.cuda.synchronize()
    elapsed, peak = time.perf_counter()-start, torch.cuda.max_memory_allocated()
    out.mkdir(exist_ok=False)
    state = dict(xyz=array(xyz), uv=array(uv), camera=array(cam), rotations=array(r), betas=array(b0),
                 translation=array(trans), objective=array(loss), initial_objective=np.array(initial_loss),
                 K=array(K), T=array(T))
    np.savez_compressed(out/"state.npz", **state)
    torch.save({"translation": trans.detach(), "global6": glob.detach(), "hand6": hand.detach(),
                "betas": b0.detach(), "optimizer": opt.state_dict(), "step": updates}, out/"optimizer.pt")
    ck = torch.load(out/"optimizer.pt", map_location="cuda")
    with torch.no_grad():
        for p, key in ((trans, "translation"), (glob, "global6"), (hand, "hand6")):
            p.copy_(ck[key])
        rebuilt = array(forward()[1])
    error = float(abs(rebuilt-state["xyz"]).max())
    require(error < 1e-7, "checkpoint reload")
    np.testing.assert_array_equal(state["betas"], initial[1])
    frozen = slice(1, None) if mode == "orientation" else slice(0, 1)
    np.testing.assert_array_equal(state["rotations"][:, frozen], initial[0][:, frozen])
    write_json(out/"runtime.json", dict(backend_s=elapsed, optimizer_steps=updates,
               forward_objective_calls=step+1, closure_calls=0, peak_allocated_gpu_bytes=peak,
               numerical_failure=failure, reload_xyz_max_abs_m=error, trajectory=trace,
               timing_scope="MANO-only serial backend; no teacher; excludes I/O/reload"))
    require(failure is None and (state["camera"][..., 2] > config["minimum_depth_m"]).all(),
            "numerical or positive-depth failure; retain saved evidence")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--config", type=Path, default=ROOT/"configs/pose_preserving_v1.json")
    a = p.parse_args()
    extra = json.loads(a.config.read_text())
    source = ROOT/extra["source_run"]
    out = ROOT/"artifacts"/a.run_id
    out.mkdir(exist_ok=False)
    config = {**json.loads((source/"config.json").read_text()), **extra}
    write_json(out/"config.json", config)
    write_json(out/"source_provenance.json", {"source_run": str(source), "teacher_inference_calls": 0,
               "source_manifest_sha256": sha256(source/"sha256.json"),
               "input_manifest_sha256": sha256(source/"input_manifest.json"),
               "preregistration_sha256": sha256(a.config)})
    shutil.copy2(source/"input_manifest.json", out/"input_manifest.json")
    for name in ("run_pose_preserving_e2.py", "run_real_e2.py", "run_mano_backend.py", "real_e2_data.py"):
        shutil.copy2(ROOT/"scripts"/name, out/name)
    originals = json.loads((source/"sha256.json").read_text())
    require(all(sha256(source/name) == h for name, h in originals.items()), "source E2 hash mismatch")
    torch, MANO, _, _, _, _, CfgNode, _ = torch_setup()
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = CfgNode.load_cfg((ROOT/"inputs/model_config.yaml").read_text())
    cfg.MANO.DATA_DIR, cfg.MANO.MODEL_PATH = str(ROOT/"inputs/data"), str(ROOT/"inputs/data/mano")
    cfg.MANO.MEAN_PARAMS = str(ROOT/"inputs/data/mano_mean_params.npz")
    mano = MANO(**{k.lower(): v for k, v in dict(cfg.MANO).items()}).cuda().eval().requires_grad_(False)
    folders = sorted(source.glob("sample_*"))
    require(len(folders) == 32, "exact 32 cached times required")
    status = []
    try:
        for i, original in enumerate(folders):
            folder = out/original.name
            folder.mkdir()
            row = json.loads((original/"sample.json").read_text())
            require(row["handedness"] == "right", "this preregistration uses the fixed right-hand cohort")
            for name in ("sample.json", "evaluation.npz"):
                shutil.copy2(original/name, folder/name)
            target = np.load(original/"evaluation.npz")
            initial = tuple(target[k] for k in ("rotations", "betas", "translation"))
            raw = np.load(original/"E2-0/state.npz")
            for variant in config["variants"]:
                dest = folder/variant["id"]
                if "reuse" in variant:
                    shutil.copytree(original/variant["reuse"], dest)
                    write_json(dest/"reuse.json", {"source": str(original/variant["reuse"]), "new_updates": 0})
                elif variant["variables"] == "translation":
                    original_solve(mano, initial, target["K"], target["T"], target["pseudo"],
                                   np.ones((2, 21)), np.ones(2), variant, config, "right", dest)
                else:
                    solve_anchor(mano, initial, target["K"], target["T"], target["pseudo"], variant, config, dest)
                np.savez_compressed(dest/"initial.npz", **{k: raw[k] for k in
                                    ("xyz", "rotations", "betas", "translation", "K", "T")})
                final = np.load(dest/"state.npz")
                require((final["camera"][..., 2] > config["minimum_depth_m"]).all(), "positive depth")
                np.testing.assert_array_equal(final["K"], raw["K"])
                np.testing.assert_array_equal(final["T"], raw["T"])
                np.testing.assert_array_equal(final["betas"], raw["betas"])
                if variant["variables"] in ("none", "translation"):
                    np.testing.assert_array_equal(final["rotations"], raw["rotations"])
                elif variant["variables"] == "orientation":
                    np.testing.assert_array_equal(final["rotations"][:, 1:], raw["rotations"][:, 1:])
                elif variant["variables"] == "fingers":
                    np.testing.assert_array_equal(final["rotations"][:, :1], raw["rotations"][:, :1])
                status.append({"sample": i, "sample_id": row["sample_id"], "variant": variant["id"],
                               "status": "reused" if "reuse" in variant else "completed"})
                print(json.dumps(status[-1]), flush=True)
            write_json(out/"matrix_status.json", status)
        write_json(out/"status.json", {"completed": True, "samples": 32, "new_optimized_states": 256,
                   "reused_states": 96, "teacher_inference_calls": 0, "E3_started": False})
    except Exception:
        (out/"failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        seal(out)


if __name__ == "__main__":
    main()
