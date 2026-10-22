"""Independently reconstruct saved optimized MANO states and matrix invariants."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from run_real_e2 import torch_setup


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    torch, MANO, _, _, _, decode, CfgNode, _ = torch_setup()
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    cfg = CfgNode.load_cfg((root/"inputs/model_config.yaml").read_text())
    cfg.MANO.DATA_DIR = str(root/"inputs/data")
    cfg.MANO.MODEL_PATH = str(root/"inputs/data/mano")
    cfg.MANO.MEAN_PARAMS = str(root/"inputs/data/mano_mean_params.npz")
    mano = MANO(**{k.lower(): v for k, v in dict(cfg.MANO).items()}).cpu().eval().requires_grad_(False)
    config = json.loads((a.run/"config.json").read_text())
    variants = {v["id"]: v for v in config["variants"]}
    verified, maximum = [], 0.
    for sample in sorted(a.run.glob("sample_*")):
        initial = np.load(sample/"evaluation.npz")
        row = json.loads((sample/"sample.json").read_text())
        for label, variant in variants.items():
            path = sample/label/"state.npz"
            if not path.exists():
                continue
            state = np.load(path)
            runtime = json.loads((path.parent/"runtime.json").read_text())
            checkpoint = torch.load(path.parent/"optimizer.pt", map_location="cpu")
            mode = variant["variables"]
            glob = (decode(checkpoint["global6"]).reshape(1, 1, 3, 3) if mode in ("orientation", "pose", "shape")
                    else torch.tensor(initial["rotations"][:, :1], dtype=torch.float32))
            hand = (decode(checkpoint["hand6"]).reshape(1, 15, 3, 3) if mode in ("pose", "shape")
                    else torch.tensor(initial["rotations"][:, 1:], dtype=torch.float32))
            beta = checkpoint["betas"] if mode == "shape" else torch.tensor(initial["betas"], dtype=torch.float32)
            with torch.no_grad():
                predicted = mano(global_orient=glob, hand_pose=hand, betas=beta, pose2rot=False).joints[0]
                if row["handedness"] == "left":
                    predicted[:, 0] *= -1
                predicted += checkpoint["translation"][0]
            error = float(np.max(abs(predicted.numpy()-state["xyz"])))
            np.testing.assert_allclose(predicted.numpy(), state["xyz"], atol=1e-6, rtol=1e-5)
            maximum = max(maximum, error)
            np.testing.assert_allclose(state["K"], initial["K"], atol=1e-5, rtol=1e-6)
            np.testing.assert_allclose(state["T"], initial["T"], atol=1e-6)
            expected_steps = 0 if mode == "none" else config["steps"]
            if runtime["numerical_failure"] is None:
                assert runtime["optimizer_steps"] == checkpoint["step"] == expected_steps
                assert runtime["forward_objective_calls"] == expected_steps+1
            assert runtime["closure_calls"] == 0
            assert runtime["reload_xyz_max_abs_m"] < 1e-7
            if mode not in ("pose", "shape"):
                np.testing.assert_array_equal(state["rotations"][:, 1:], initial["rotations"][:, 1:])
            if mode in ("translation", "none"):
                np.testing.assert_array_equal(state["rotations"], initial["rotations"])
            if mode != "shape":
                np.testing.assert_array_equal(state["betas"], initial["betas"])
            if mode == "none":
                np.testing.assert_allclose(state["translation"], initial["translation"], atol=1e-7)
            assert len(runtime["trajectory"]) == (1 if expected_steps == 0 else expected_steps//config["trace_interval"]+1)
            verified.append({"sample_id": row["sample_id"], "variant": label,
                             "cpu_reconstruction_max_abs_m": error, "optimizer_steps": expected_steps})
    a.output.write_text(json.dumps({"passed": True, "states": len(verified), "rows": verified,
                                   "maximum_cpu_vs_gpu_reconstruction_m": maximum,
                                   "checks": ["checkpoint MANO reconstruction", "camera immutability",
                                              "frozen parameters", "optimizer budgets", "reload consistency",
                                              "compact trajectory length"],
                                   "optimizer_updates_performed": 0}, indent=2)+"\n")
    print(json.dumps({"passed": True, "states": len(verified), "maximum_error_m": maximum}), flush=True)


if __name__ == "__main__":
    main()
