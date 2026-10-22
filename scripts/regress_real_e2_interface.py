"""Read-only regression against COMPLETED synthetic E2 arrays. No optimizer steps."""
import argparse
import json
from pathlib import Path
import shutil
import traceback
import numpy as np
from real_e2_data import validate_geometry, crop_affine, loss_numpy, sha256, require
from run_real_e2 import project_torch, reprojection_torch, torch_setup, seal, write_json
from evaluate_real_e2 import evaluate

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    require(Path(args.run_id).name == args.run_id and args.run_id not in (".", ".."), "invalid run ID")
    out = ROOT/"artifacts"/args.run_id
    out.mkdir(exist_ok=False)
    for name in ("real_e2_data.py", "run_real_e2.py", "evaluate_real_e2.py", Path(__file__).name):
        shutil.copy2(ROOT/"scripts"/name, out/name)
    config = json.loads((ROOT/"configs/real_e2_v1.json").read_text())
    write_json(out/"config.json", config)
    before = {p.name: sha256(p) for p in args.source.iterdir() if p.is_file()}
    write_json(out/"source_hashes.json", before)
    checks = []
    try:
        torch, MANO, tensor, array, geometry, decode, CfgNode, _ = torch_setup()
        torch.set_num_threads(4)
        torch.manual_seed(42)
        data = np.load(args.source/"inputs.npz")
        T = np.tile(np.eye(4), (2, 1, 1))
        T[:, :3, :3], T[:, :3, 3] = data["R"], data["ext_t"]
        reports = []
        for i in range(len(data["gt"])):
            reports.append(validate_geometry(data["K"][i], T, data["gt"][i], data["clean"][i],
                                             np.ones((2, 21), bool), tolerance=.003))
        checks.append("eight existing fixtures: K, extrinsics, GT projections, baseline, stereo cheirality")
        def rejected(name, **changes):
            values = dict(K=data["K"][0], T=T, xyz=data["gt"][0], uv=data["clean"][0],
                          valid=np.ones((2, 21), bool), tolerance=.003)
            values.update(changes)
            try:
                validate_geometry(**values)
            except ValueError:
                checks.append(name)
                return
            raise AssertionError(f"did not reject {name}")
        rejected("rejected inverse extrinsic direction", T=np.linalg.inv(T))
        collapsed = T.copy()
        collapsed[1] = collapsed[0]
        rejected("rejected zero baseline", T=collapsed)
        rejected("rejected millimeter data labeled meters", xyz=data["gt"][0]*1000)
        rejected("rejected mismatched synchronized annotation", uv=data["clean"][1])
        bad_K = data["K"][0].copy()
        bad_K[:, :2] *= .5
        rejected("rejected unscaled image calibration", K=bad_K)
        for side in ("left", "right"):
            A = crop_affine([103, 72, 303, 352], side)
            points = np.c_[data["clean"][0, 0], np.ones(21)]
            np.testing.assert_allclose(points @ A.T @ np.linalg.inv(A).T, points, atol=1e-10)
            require(np.sign(np.linalg.det(A[:2, :2])) == (-1 if side == "left" else 1), "mirror count")
        checks.append("left/right crop roundtrip and exactly one left reflection")
        xyz = tensor(data["gt"][0]).requires_grad_(True)
        uv, camera = project_torch(xyz, tensor(data["K"][0]), tensor(T))
        np.testing.assert_allclose(array(uv), data["clean"][0], atol=.0001)
        for robust in (False, True):
            for conf in (np.ones((2, 21)), data["confidence"][0]):
                expected = loss_numpy(array(uv), data["noisy"][0], conf, [1, 1], robust)
                actual = reprojection_torch(uv, tensor(data["noisy"][0]), tensor(conf),
                                            tensor([1, 1]), robust, 3.)
                np.testing.assert_allclose(float(actual.detach()), expected, rtol=1e-5, atol=1e-5)
                gradient = torch.autograd.grad(actual, xyz, retain_graph=True)[0]
                require(torch.isfinite(gradient).all(), "nonfinite loss gradient")
        checks.append("L2/Huber uniform/oracle fixture weights: independent reference and finite gradients")
        duplicate_uv = uv[:1].repeat(2, 1, 1)
        duplicate_target = tensor(data["noisy"][0, :1]).repeat(2, 1, 1)
        for robust in (False, True):
            a = reprojection_torch(duplicate_uv, duplicate_target, tensor(np.ones((2, 21))),
                                  tensor([1, 0]), robust, 3.)
            b = reprojection_torch(duplicate_uv, duplicate_target, tensor(np.ones((2, 21))),
                                  tensor([1, 1]), robust, 3.)
            np.testing.assert_allclose(array(a), array(b), rtol=1e-6)
        checks.append("duplicating identical view does not double objective scale")
        equal = reprojection_torch(uv, tensor(data["noisy"][0]), tensor(np.ones((2, 21))),
                                   tensor([1, 1]), False, 3.)
        downweighted = reprojection_torch(uv, tensor(data["noisy"][0]), tensor(np.ones((2, 21))),
                                          tensor([1, .1]), False, 3.)
        require(float(downweighted.detach()) < float(equal.detach()), "view confidence canceled by normalization")
        checks.append("per-view confidence survives joint normalization and downweights corrupted right view")
        conf = np.ones((2, 21))
        conf[1] = 0
        masked_target = data["noisy"][0].copy()
        masked_target[1] = np.nan
        a = reprojection_torch(uv, tensor(masked_target), tensor(conf), tensor([1, 1]), True, 3.)
        require(torch.isfinite(a), "masked invalid view contaminates loss")
        checks.append("all-masked view excluded from view denominator and NaN residual")
        cfg = CfgNode.load_cfg((ROOT/"inputs/model_config.yaml").read_text())
        cfg.MANO.DATA_DIR = str(ROOT/"inputs/data")
        cfg.MANO.MODEL_PATH = str(ROOT/"inputs/data/mano")
        cfg.MANO.MEAN_PARAMS = str(ROOT/"inputs/data/mano_mean_params.npz")
        mano = MANO(**{k.lower(): v for k, v in dict(cfg.MANO).items()}).cuda().eval()
        mano.requires_grad_(False)
        cached = np.load(args.source/"stereo_robust_final.npz")
        checkpoint = torch.load(args.source/"stereo_robust_optimizer.pt", map_location="cuda")
        r = torch.cat([decode(checkpoint["global6"]).reshape(-1, 1, 3, 3),
                       decode(checkpoint["hand6"]).reshape(-1, 15, 3, 3)], 1)
        with torch.no_grad():
            restored = geometry(mano, r, tensor(data["betas"])) + checkpoint["translation"][:, None]
        reload_error = float(np.max(abs(array(restored)-cached["xyz"])))
        require(reload_error < 1e-6, "existing checkpoint MANO reload mismatch")
        checks.append("existing optimized checkpoint reloaded to cached MANO XYZ; no updates")
        original_report = json.loads((args.source/"results.json").read_text())["results"]["stereo_robust"]
        metrics = []
        for i in range(len(data["gt"])):
            state = {k: cached[k][i] for k in ("xyz", "uv", "objective", "initial_objective")}
            state.update({k: cached[k][i:i+1] for k in ("rotations", "betas", "translation")})
            target = {"K": data["K"][i], "T": T, "gt": data["gt"][i], "gt_uv": data["clean"][i],
                      "valid": np.ones((2, 21), bool), "affine": np.tile(np.eye(3), (2, 1, 1)),
                      "pseudo": data["noisy"][i],
                      **{k: data[k][i:i+1] for k in ("rotations", "betas", "translation")}}
            runtime = {"backend_s": 0., "optimizer_steps": 0, "closure_calls": 0,
                       "forward_objective_calls": 0, "peak_allocated_gpu_bytes": 0, "numerical_failure": None}
            metrics.append(evaluate(state, target, runtime, config))
        for key in ("absolute_mpjpe_mm", "root_relative_mpjpe_mm", "wrist_depth_error_mm"):
            np.testing.assert_allclose(np.mean([m[key] for m in metrics]), original_report["mean"][key],
                                       atol=1e-4, rtol=1e-5)
        checks.append("new independent metric evaluator matches three archived core GT metrics")
        after = {p.name: sha256(p) for p in args.source.iterdir() if p.is_file()}
        require(before == after, "source artifacts modified")
        checks.append("every inherited fixture artifact unchanged by SHA256")
        write_json(out/"regression.json", {"passed": True, "checks": checks, "fixture_rows": reports,
                   "reload_max_abs_m": reload_error, "optimizer_updates": 0,
                   "new_synthetic_experiments": 0, "real_e2_executed": False,
                   "scope": "geometry, loss, crop, checkpoint and metric interfaces only; no RGB inference/real data qualification"})
        print(json.dumps({"passed": True, "checks": len(checks), "optimizer_updates": 0,
                          "real_e2_executed": False, "output": str(out)}), flush=True)
    except Exception:
        (out/"failure.txt").write_text(traceback.format_exc())
        write_json(out/"regression.json", {"passed": False, "completed_checks": checks,
                   "optimizer_updates": 0, "real_e2_executed": False})
        raise
    finally:
        seal(out)


if __name__ == "__main__":
    main()
