"""Real right-frame ±1 sensitivity; reuse main E2-4 S0 without rerunning it."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import traceback
import numpy as np
from real_e2_data import load_manifest, read_view, sha256, require
from run_real_e2 import torch_setup, infer_teacher, solve, seal, write_json
from prepare_dexycb_real_e2 import official_subset
from evaluate_real_e2 import evaluate_run

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--main-run", type=Path, required=True)
    p.add_argument("--preparation", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    a = p.parse_args()
    require(json.loads((a.main_run/"status.json").read_text())["real_e2_executed"], "main E2 must finish first")
    out = ROOT/"artifacts"/a.run_id
    out.mkdir(exist_ok=False)
    for name in ("run_dexycb_sync_sensitivity.py", "run_real_e2.py", "real_e2_data.py",
                 "evaluate_real_e2.py", "prepare_dexycb_real_e2.py"):
        shutil.copy2(ROOT/"scripts"/name, out/name)
    config = json.loads((a.main_run/"config.json").read_text())
    variant = next(v for v in config["variants"] if v["id"] == "E2-4")
    config["variants"] = [{**variant, "id": "E2-S1"}, {**variant, "id": "E2-S2"}]
    config["synchronization"] = "right frame -1/+1; original synchronized GT and identical initial state"
    write_json(out/"config.json", config)
    manifest_path = a.preparation/"manifest_reviewed.json"
    shutil.copy2(manifest_path, out/"original_synchronous_manifest.json")
    _, samples, _ = load_manifest(manifest_path, config)
    dataset = official_subset(a.data.resolve(), out)
    detections = {int(k): v for k, v in json.loads((a.preparation/"detector_predictions.json").read_text()).items()}
    lookup = {(v["sequence"], v["camera"], v["frame"]): (k, v) for k, v in detections.items()}
    selected, excluded = [], []
    for i, sample in enumerate(samples):
        row = sample["row"]
        keys = [(row["sequence_id"], row["views"][1]["camera_serial"], row["frame_index"]+delta)
                for delta in (-1, 1)]
        if all(key in lookup and lookup[key][1]["selected"] is not None for key in keys):
            selected.append((i, sample, keys))
        else:
            excluded.append({"sample_id": row["sample_id"], "reason": "neighbor absent or detector missed"})
    write_json(out/"selection.json", {"selected": [s["row"]["sample_id"] for _, s, _ in selected],
               "excluded": excluded, "selection": "common S0/S-1/S+1 neighboring-frame detector availability only"})
    torch, _, tensor, array, _, _, CfgNode, HAMER = torch_setup()
    torch.manual_seed(config["seed"])
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = CfgNode.load_cfg((ROOT/"inputs/model_config.yaml").read_text())
    cfg.MANO.DATA_DIR = str(ROOT/"inputs/data")
    cfg.MANO.MODEL_PATH = str(ROOT/"inputs/data/mano")
    cfg.MANO.MEAN_PARAMS = str(ROOT/"inputs/data/mano_mean_params.npz")
    cfg.MODEL.BBOX_SHAPE = [192, 256]
    cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS", None)
    model = HAMER(cfg, init_renderer=False)
    provenance = json.loads((a.main_run/"teacher_provenance.json").read_text())
    require(sha256(ROOT/"inputs/hamer.ckpt") == provenance["checkpoint_sha256"],
            "teacher checkpoint changed after main run")
    write_json(out/"teacher_provenance.json", provenance)
    checkpoint = torch.load(ROOT/"inputs/hamer.ckpt", map_location="cpu")
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
    require(not any(not k.startswith(("discriminator.", "renderer.", "mesh_renderer."))
                    for k in incompatible.missing_keys+incompatible.unexpected_keys), "checkpoint mismatch")
    del checkpoint
    model = model.cuda().eval().requires_grad_(False)
    status = []
    try:
        for n, (i, sample, keys) in enumerate(selected):
            folder = out/f"sample_{n:03d}"
            folder.mkdir()
            write_json(folder/"sample.json", sample["row"])
            source = a.main_run/f"sample_{i:03d}"
            targets = dict(np.load(source/"evaluation.npz"))
            initial = tuple(targets[k] for k in ("rotations", "betas", "translation"))
            # Same synchronous GT for both offsets; pseudo targets are variant-specific.
            np.savez_compressed(folder/"evaluation.npz", **targets)
            for key, v in zip(keys, config["variants"]):
                idx, detection = lookup[key]
                official_entry = dataset[idx]
                view = copy.deepcopy(sample["row"]["views"][1])
                view["image"] = official_entry["color_file"]
                view["image_sha256"] = sha256(view["image"])
                uv = np.asarray(detection["selected"]["uv"])
                center = (uv.min(0)+uv.max(0))/2
                extent = np.max(uv.max(0)-uv.min(0))*2
                view["crop_box_xyxy"] = np.r_[center-extent/2, center+extent/2].tolist()
                write_json(folder/f"{v['id']}_neighbor_input.json", {
                    "sequence": key[0], "camera_serial": key[1], "frame_index": key[2],
                    "image": view["image"], "image_sha256": view["image_sha256"],
                    "crop_box_xyxy": view["crop_box_xyxy"], "crop_source": "predicted_detector",
                    "official_toolkit_index": idx, "evaluation_gt": "original synchronous time",
                    "detector_provenance": str((a.preparation/"detector_provenance.json").resolve())})
                processed = read_view(view, ROOT, sample["row"]["handedness"])
                neighbor = {"row": sample["row"], "views": [processed]}
                teacher = infer_teacher(model, cfg, neighbor, tensor, array, config)[0]
                np.savez_compressed(folder/f"{v['id']}_neighbor_teacher.npz", **teacher)
                pseudo = targets["pseudo"].copy()
                pseudo[1] = teacher["pseudo"]
                result = solve(model.mano, initial, targets["K"], targets["T"], pseudo,
                               np.ones((2, 21)), np.ones(2), v, config, sample["row"]["handedness"],
                               folder/v["id"])
                # Independent evaluation reads these targets for residual only; GT unchanged.
                np.savez_compressed(folder/v["id"]/"evaluation.npz", **{**targets, "pseudo": pseudo})
                status.append({"sample_id": sample["row"]["sample_id"], "variant": v["id"],
                               "right_frame": key[2], "right_image": view["image"], "status": "completed"})
                print(json.dumps(status[-1]), flush=True)
            write_json(out/"matrix_status.json", status)
        evaluate_run(out)
        write_json(out/"status.json", {"completed": True, "samples": len(selected),
                   "S0_source": str(a.main_run.resolve()), "S0_variant": "E2-4",
                   "S0_rerun": False, "offset_frame_nominal_ms": 1000/30})
    except Exception:
        (out/"failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        seal(out)


if __name__ == "__main__":
    main()
