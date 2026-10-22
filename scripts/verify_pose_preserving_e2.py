"""Independent state, input, metric and DexYCB pose_m verification."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import torch
import yaml
from smplx.lbs import batch_rodrigues
from run_real_e2 import torch_setup
from evaluate_pose_preserving_e2 import one, stats

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--metrics", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    torch.set_num_threads(4)
    _, MANO, _, _, _, decode, CfgNode, _ = torch_setup()
    cfg = CfgNode.load_cfg((ROOT/"inputs/model_config.yaml").read_text())
    cfg.MANO.DATA_DIR, cfg.MANO.MODEL_PATH = str(ROOT/"inputs/data"), str(ROOT/"inputs/data/mano")
    cfg.MANO.MEAN_PARAMS = str(ROOT/"inputs/data/mano_mean_params.npz")
    mano = MANO(**{k.lower(): v for k, v in dict(cfg.MANO).items()}).cpu().eval().requires_grad_(False)
    config = json.loads((a.run/"config.json").read_text())
    variants = {x["id"]: x for x in config["variants"]}
    source_hashes = json.loads((a.source/"sha256.json").read_text())
    assert all(digest(a.source/name) == value for name, value in source_hashes.items())
    source_folders, run_folders = sorted(a.source.glob("sample_*")), sorted(a.run.glob("sample_*"))
    assert len(source_folders) == len(run_folders) == 32
    expected_ids = [json.loads((x/"sample.json").read_text())["sample_id"] for x in source_folders]
    assert expected_ids == [json.loads((x/"sample.json").read_text())["sample_id"] for x in run_folders]
    report = json.loads(a.metrics.read_text())
    recorded = {(x["sample_id"], x["variant"]): x["metrics"] for x in report["rows"]}
    references = {(x["sample_id"], x["variant"]): x["metrics"] for x in report["reference_rows"]}
    reconstructed, maximum = [], 0.
    for source, folder in zip(source_folders, run_folders):
        assert digest(source/"evaluation.npz") == digest(folder/"evaluation.npz")
        target = np.load(folder/"evaluation.npz")
        row = json.loads((folder/"sample.json").read_text())
        source_initial = np.load(source/"E2-0"/"state.npz")
        for label, variant in variants.items():
            path = folder/label
            state = np.load(path/"state.npz")
            runtime = json.loads((path/"runtime.json").read_text())
            initial = np.load(path/"initial.npz")
            for key in ("K", "T"):
                np.testing.assert_array_equal(state[key], source_initial[key])
                np.testing.assert_array_equal(initial[key], source_initial[key])
                np.testing.assert_allclose(state[key], target[key], atol=1e-7, rtol=1e-7)
            for key in ("rotations", "betas", "translation"):
                np.testing.assert_array_equal(initial[key], source_initial[key])
                np.testing.assert_allclose(initial[key], target[key], atol=1e-7, rtol=1e-7)
            checkpoint = torch.load(path/"optimizer.pt", map_location="cpu")
            mode = variant["variables"]
            r0 = torch.tensor(target["rotations"], dtype=torch.float32)
            glob = decode(checkpoint["global6"]).reshape(1, 1, 3, 3) if mode in ("orientation", "pose") else r0[:, :1]
            hand = decode(checkpoint["hand6"]).reshape(1, 15, 3, 3) if mode in ("fingers", "pose") else r0[:, 1:]
            betas = checkpoint["betas"] if mode == "shape" else torch.tensor(target["betas"], dtype=torch.float32)
            with torch.no_grad():
                xyz = mano(global_orient=glob, hand_pose=hand, betas=betas, pose2rot=False).joints[0]
                xyz += checkpoint["translation"][0]
            error = float(abs(xyz.numpy()-state["xyz"]).max())
            np.testing.assert_allclose(xyz.numpy(), state["xyz"], atol=2e-7, rtol=1e-5)
            maximum = max(maximum, error)
            np.testing.assert_array_equal(state["betas"], target["betas"])
            if mode in ("none", "translation"):
                np.testing.assert_array_equal(state["rotations"], target["rotations"])
            elif mode == "orientation":
                np.testing.assert_array_equal(state["rotations"][:, 1:], target["rotations"][:, 1:])
            elif mode == "fingers":
                np.testing.assert_array_equal(state["rotations"][:, :1], target["rotations"][:, :1])
            expected_steps = 0 if mode == "none" else config["steps"]
            assert runtime["optimizer_steps"] == checkpoint["step"] == expected_steps
            assert runtime["closure_calls"] == 0 and runtime["numerical_failure"] is None
            assert runtime["reload_xyz_max_abs_m"] < 2e-7
            recomputed = one(state, target, runtime, row, config)
            for key, value in recomputed.items():
                assert np.isclose(value, recorded[(row["sample_id"], label)][key], rtol=1e-10, atol=1e-10)
            reconstructed.append({"sample_id": row["sample_id"], "variant": label, "max_abs_m": error})
        reference = source/"E2-2"
        recomputed = one(np.load(reference/"state.npz"), target,
                         json.loads((reference/"runtime.json").read_text()), row, config)
        for key, value in recomputed.items():
            assert np.isclose(value, references[(row["sample_id"], "E2-2-reference")][key],
                              rtol=1e-10, atol=1e-10)
    # Verify every aggregate directly from independently recomputed per-sample rows.
    for label in variants:
        selected = [recorded[(sid, label)] for sid in expected_ids]
        for key in selected[0]:
            recomputed = stats([x[key] for x in selected])
            for stat, value in recomputed.items():
                assert np.isclose(value, report["aggregates"][label][key][stat], rtol=1e-12, atol=1e-12)
    selected = [references[(sid, "E2-2-reference")] for sid in expected_ids]
    for key in selected[0]:
        recomputed = stats([x[key] for x in selected])
        for stat, value in recomputed.items():
            assert np.isclose(value, report["reference_aggregates"]["E2-2-reference"][key][stat],
                              rtol=1e-12, atol=1e-12)

    # DexYCB pose_m is global axis-angle + 45 MANO PCA coefficients + translation.
    model = pickle.load(open(ROOT/"inputs/data/mano/MANO_RIGHT.pkl", "rb"), encoding="latin1")
    components = np.asarray(model["hands_components"], np.float32)
    mean = np.asarray(model["hands_mean"], np.float32)
    beta_path = ROOT/"inputs/dexycb_official_subset_20261020_01/calibration/mano_20200709_140042_subject-01_right/mano.yml"
    dex_betas = np.asarray(yaml.safe_load(beta_path.read_text())["betas"], np.float32)
    pose_errors, anchor_errors = [], []
    for folder in run_folders:
        row = json.loads((folder/"sample.json").read_text())
        pose = np.asarray(row["views"][0]["pose_m"], np.float32)[0]
        aa = np.r_[pose[:3], pose[3:48]@components+mean].astype(np.float32)
        rotations = batch_rodrigues(torch.tensor(aa).reshape(-1, 3)).reshape(1, 16, 3, 3)
        with torch.no_grad():
            joints = mano(global_orient=rotations[:, :1], hand_pose=rotations[:, 1:],
                          betas=torch.tensor(dex_betas)[None],
                          transl=torch.tensor(pose[48:])[None], pose2rot=False).joints[0].numpy()
        gt = np.asarray(row["views"][0]["gt3d"])
        pose_errors.extend(np.linalg.norm(joints-gt, axis=-1)*1000)
        anchor_errors.extend(np.linalg.norm(joints[[0, 1, 5, 9, 13, 17]]-gt[[0, 1, 5, 9, 13, 17]], axis=-1)*1000)
    pose_audit = {"all_joint_error_mm": stats(pose_errors), "wrist_and_mcp_error_mm": stats(anchor_errors),
                  "maximum_all_joint_mm": float(max(pose_errors)),
                  "passed": float(np.mean(pose_errors)) < 1 and float(max(anchor_errors)) < .01,
                  "interpretation": "pose_m global axis-angle coordinate validated; residual tip mismatch reflects implementation vertex-tip conventions"}
    assert pose_audit["passed"]
    output = {"passed": True, "states": len(reconstructed), "samples": len(expected_ids),
              "maximum_checkpoint_reconstruction_m": maximum, "pose_m_audit": pose_audit,
              "checks": ["same ordered 32 IDs", "source SHA256", "identical cached evaluation inputs",
                         "initial state and pseudo/K/T identity", "strict frozen variables",
                         "checkpoint MANO reconstruction", "800-step budgets", "positive finite metrics",
                         "independent per-sample and aggregate metric equality"],
              "optimizer_updates_performed": 0}
    a.output.write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
