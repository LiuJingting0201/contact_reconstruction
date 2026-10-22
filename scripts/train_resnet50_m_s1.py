"""M-S1: frozen subject-disjoint real dev, full mixed training, no KD."""
import argparse
from collections import Counter
import hashlib
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

import train_resnet50_real_mixed as base

ROOT = base.ROOT
DOMAINS = ("dexycb", "freihand", "rhd")
FINGERS = ("thumb", "index", "middle", "ring", "little")


class FrozenDex(base.DexYCBDataset):
    def __init__(self, root, size, augmentation, train):
        self.root, self.image_size = root, size
        self.augmentation, self.train = augmentation, train
        self.manifest = json.loads((root / "sealed_manifest.json").read_text())
        assert self.manifest["subject"] == (
            "20200813-subject-02" if train else "20200928-subject-07")
        assert self.manifest["split"] == ("train" if train else "dev")
        self.rows = [
            {**s, **v, "views": None} for s in self.manifest["samples"]
            for v in s["views"]
        ]


def datasets(config, split):
    paths = config["datasets"]
    size, aug = config["model"]["image_size"], config["augmentation"]
    train = split == "train"
    frozen = json.loads((ROOT / config["split_dir"] / "freihand_indices.json").read_text())
    return {
        "dexycb": FrozenDex(ROOT / paths[f"dexycb_{split}"], size, aug, train),
        "freihand": Subset(base.FreiHANDDataset(
            ROOT / paths["freihand"], size, aug, train, versions=4), frozen[split]),
        "rhd": base.RHDDataset(ROOT / paths[f"rhd_{split}"], ROOT / paths["rhd_images"],
                               size, aug, train),
    }


def freeze(config):
    """Only labels and IDs; no model construction, no E4 image or label reads."""
    out = ROOT / config["split_dir"]
    out.mkdir(parents=True, exist_ok=False)
    paths = config["datasets"]
    captures = {"train": [], "dev": []}
    for i in range(32560):
        key = hashlib.sha256(f"freihand-capture-{i}-20261021".encode()).hexdigest()
        captures["dev" if int(key[:8], 16) % 10 == 0 else "train"].append(i)
    indices = {split: [i + v * 32560 for v in range(4) for i in ids]
               for split, ids in captures.items()}
    base.write_json(out / "freihand_indices.json", indices)
    subjects, sequences, split_files, data_hashes = {}, {}, {}, {}
    for split in ("train", "dev", "final_e4"):
        rows = []
        subjects[split], sequences[split] = set(), set()
        roots = ([ROOT / paths[f"dexycb_{split}"]] if split != "final_e4"
                 else [ROOT / p for p in config["final_manifest_roots"]])
        for root in roots:
            manifest_path = root / "sealed_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            data_hashes[str(manifest_path.relative_to(ROOT))] = base.sha256(manifest_path)
            for s in manifest["samples"]:
                subjects[split].add(s["subject"])
                sequences[split].add(s["sequence"])
                for v in s["views"]:
                    rows.append({
                        "dataset": "dexycb", "sample_id": s["sample_id"] + ":" + v["camera_serial"],
                        "subject": s["subject"], "sequence": s["sequence"],
                        "frame": s["frame_index"], "image": str(root.relative_to(ROOT) / v["image"]),
                        "image_sha256": manifest["extracted_files"][v["image"]]["sha256"],
                        "label_sha256": manifest["extracted_files"][v["label"]]["sha256"],
                        "source_manifest_sha256": data_hashes[str(manifest_path.relative_to(ROOT))],
                    })
        if split != "final_e4":
            frei = ROOT / paths["freihand"]
            labels_hash = {}
            for name in ("K", "xyz", "mano", "scale"):
                f = frei / f"training_{name}.json"
                labels_hash[name] = base.sha256(f)
                data_hashes[str(f.relative_to(ROOT))] = labels_hash[name]
            for i in indices[split]:
                image = frei / "training/rgb" / f"{i:08d}.jpg"
                mask = frei / "training/mask" / f"{i % 32560:08d}.jpg"
                rows.append({"dataset": "freihand", "sample_id": f"freihand:{i%32560:05d}:rgb{i:08d}",
                             "subject": None, "sequence": None, "frame": i % 32560,
                             "capture_group": i % 32560, "background_version": i // 32560,
                             "image": str(image.relative_to(ROOT)), "image_sha256": base.sha256(image),
                             "mask": str(mask.relative_to(ROOT)), "mask_sha256": base.sha256(mask),
                             "annotation_sha256": labels_hash})
            rhd_path = ROOT / paths[f"rhd_{split}"]
            data_hashes[str(rhd_path.relative_to(ROOT))] = base.sha256(rhd_path)
            with np.load(rhd_path, allow_pickle=False) as archive:
                raw = {name: archive[name] for name in ("sample_id", "image", "image_id")}
            image_hashes = {}
            for i, sid in enumerate(raw["sample_id"]):
                image = ROOT / paths["rhd_images"] / str(raw["image"][i])
                key = str(image.relative_to(ROOT))
                if key not in image_hashes:
                    image_hashes[key] = base.sha256(image)
                rows.append({"dataset": "rhd", "sample_id": str(sid), "subject": None,
                             "sequence": str(raw["image"][i]).split("/")[0],
                             "frame": int(raw["image_id"][i]), "image": key,
                             "image_sha256": image_hashes[key],
                             "annotation_sha256": data_hashes[str(rhd_path.relative_to(ROOT))]})
        for row in rows:
            row["sha256"] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        path = out / f"{split}.jsonl"
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        split_files[path.name] = {"sha256": base.sha256(path), "count": len(rows),
                                 "domains": {d: sum(r["dataset"] == d for r in rows) for d in DOMAINS}}
    for a, b in (("train", "dev"), ("train", "final_e4"), ("dev", "final_e4")):
        assert not subjects[a] & subjects[b]
        assert not sequences[a] & sequences[b]
    assert not set(captures["train"]) & set(captures["dev"])
    rhd_train = np.load(ROOT / paths["rhd_train"])
    rhd_dev = np.load(ROOT / paths["rhd_dev"])
    assert not set(rhd_train["image"]) & set(rhd_dev["image"])
    data_hashes[str((out / "freihand_indices.json").relative_to(ROOT))] = base.sha256(
        out / "freihand_indices.json")
    base.write_json(out / "freeze.json", {
        "config_sha256": base.sha256(ROOT / config["_config_path"]),
        "subjects": {k: sorted(v) for k, v in subjects.items()},
        "sequences": {k: sorted(v) for k, v in sequences.items()},
        "split_files": split_files, "data_hashes": data_hashes,
        "freihand_limitation": "Capture-disjoint only; published labels do not expose "
            "subject/sequence identity. No cross-subject or sequence independence claim.",
        "final_e4_access": "Previously sealed manifests only; no image/label/model access.",
    })
    print(json.dumps(split_files), flush=True)


class SequenceMixSampler:
    """Fixed domain counts; shuffled exhaustive cycles within each Dex sequence."""
    def __init__(self, data, config):
        self.data, self.config = data, config
        self.groups = {}
        for i, row in enumerate(data["dexycb"].rows):
            # Alternate sequence/side/annotation-availability strata without replacing
            # full-data eligibility by a fixed subset.
            key = (row["sequence"], row["mano_side"], row["occlusion_bucket"])
            self.groups.setdefault(key, []).append(i)

    def __len__(self):
        return self.config["training"]["updates"]

    def __iter__(self):
        rng = np.random.default_rng(self.config["seed"])
        offsets = np.cumsum([0] + [len(self.data[d]) for d in DOMAINS[:-1]])
        pools = {("dexycb", k): np.array(v) for k, v in self.groups.items()}
        pools.update({(d, None): np.arange(len(self.data[d])) for d in DOMAINS[1:]})
        orders, positions = {}, {}
        def take(key):
            if key not in positions or positions[key] == len(orders[key]):
                orders[key] = rng.permutation(pools[key])
                positions[key] = 0
            result = int(orders[key][positions[key]])
            positions[key] += 1
            return result
        keys = list(self.groups)
        group_order, position = rng.permutation(len(keys)), 0
        for _ in range(len(self)):
            batch = []
            for d, offset in zip(DOMAINS, offsets):
                for _ in range(self.config["batch_composition"][d]):
                    if d == "dexycb":
                        if position == len(keys):
                            group_order, position = rng.permutation(len(keys)), 0
                        k = keys[int(group_order[position])]
                        position += 1
                    else:
                        k = None
                    batch.append(int(offset) + take((d, k)))
            yield rng.permutation(batch).tolist()


def stats(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"min": None, "median": None, "p95": None, "max": None}
    return dict(zip(("min", "median", "p95", "max"),
                    map(float, (x.min(), np.median(x), np.percentile(x, 95), x.max()))))


def metrics(arrays):
    p, t, uv, gt, valid, cam, depth, orient = (
        arrays[k] for k in ("xyz", "target_xyz", "uv", "target_uv", "valid", "cam", "depth", "orient"))
    finite = (np.isfinite(p).all((1, 2)) & np.isfinite(uv).all((1, 2)) &
              np.isfinite(cam).all(1) & np.isfinite(depth).all(1))
    span = np.ptp(uv, axis=1).max(1) * 256
    collapse = (span < 5) | ~np.isfinite(span)
    negative = (depth <= 0).any(1)
    result = {"count": len(p), "finite_count": int(finite.sum()),
              "nonfinite_rate": float((~finite).mean()),
              "negative_depth_rate": float(negative.mean()),
              "projection_collapse_rate": float(collapse.mean()),
              "failure_rate": float((~finite | negative | collapse).mean()),
              "projection_span_px": stats(span), "weak_camera_scale": stats(cam[:, 0])}
    p, t, uv, gt, valid = (x[finite] for x in (p, t, uv, gt, valid))
    if not len(p):
        return result
    error = np.linalg.norm(p - t, axis=-1) * 1000
    # Proper orthogonal rotation, wrist anchored, NO scale or centroid fitting.
    u, _, vt = np.linalg.svd(np.einsum("nji,njk->nik", p, t))
    sign = np.ones((len(p), 3))
    sign[:, -1] = np.where(np.linalg.det(u @ vt) < 0, -1, 1)
    rotation = (u * sign[:, None, :]) @ vt
    aligned = np.linalg.norm(p @ rotation - t, axis=-1) * 1000
    pb = np.stack([p[:, b] - p[:, a] for a, b in base.EDGES], 1)
    tb = np.stack([t[:, b] - t[:, a] for a, b in base.EDGES], 1)
    cos = (pb * tb).sum(-1) / np.maximum(
        np.linalg.norm(pb, axis=-1) * np.linalg.norm(tb, axis=-1), 1e-12)
    epe = np.linalg.norm(uv - gt, axis=-1) * 256
    go = orient[np.isfinite(orient)]
    result.update({
        "wrist_relative_mpjpe_mm": float(error.mean()),
        "rotation_aligned_mpjpe_mm": float(aligned.mean()),
        "fingertip_wrist_relative_error_mm": float(error[:, base.TIPS].mean()),
        "per_finger_error_mm": {name: float(error[:, 1 + 4*i:5 + 4*i].mean())
                                for i, name in enumerate(FINGERS)},
        "bone_direction_angular_error_deg": float(np.degrees(np.arccos(np.clip(cos, -1, 1))).mean()),
        "visible_2d_epe_px": float(epe[valid].mean()) if valid.any() else None,
        "visible_2d_count": int(valid.sum()),
        "global_orientation_error_deg": float(go.mean()) if len(go) else None,
        "global_orientation_count": len(go),
    })
    return result


def evaluate(model, loaders, device, out, tag):
    model.eval()
    result = {"tag": tag, "domains": {}}
    pooled = []
    for domain, loader in loaders.items():
        chunks = {k: [] for k in ("xyz", "target_xyz", "uv", "target_uv", "valid", "cam", "depth", "orient")}
        ids = []
        with torch.no_grad():
            for batch in loader:
                pred = model.forward_step({"img": batch["img"].to(device)})
                xyz = pred["pred_keypoints_3d"]
                orient = base.rotation_angle(
                    pred["pred_mano_params"]["global_orient"][:, 0],
                    batch["global_orient"].to(device)).cpu().numpy() * 180 / np.pi
                orient[~batch["global_orient_valid"].numpy()] = np.nan
                values = {
                    "xyz": (xyz - xyz[:, :1]).cpu().numpy(),
                    "target_xyz": batch["xyz"].numpy(),
                    "uv": pred["pred_keypoints_2d"].cpu().numpy(),
                    "target_uv": batch["uv"].numpy(), "valid": batch["uv_valid"].numpy(),
                    "cam": pred["pred_cam"].cpu().numpy(),
                    "depth": base.camera_depth(pred).cpu().numpy(), "orient": orient,
                }
                for k, v in values.items():
                    chunks[k].append(v)
                ids.extend(batch["sample_id"])
        arrays = {k: np.concatenate(v) for k, v in chunks.items()}
        pooled.append(arrays)
        np.savez_compressed(out / f"{tag}_{domain}_predictions.npz", sample_id=np.array(ids), **arrays)
        result["domains"][domain] = metrics(arrays)
        print(json.dumps({"evaluated": tag, "domain": domain, "metrics": result["domains"][domain]},
                         allow_nan=False), flush=True)
    scalar_keys = ("wrist_relative_mpjpe_mm", "rotation_aligned_mpjpe_mm",
                   "fingertip_wrist_relative_error_mm", "bone_direction_angular_error_deg",
                   "visible_2d_epe_px", "nonfinite_rate", "negative_depth_rate",
                   "projection_collapse_rate", "failure_rate")
    result["mixed_aggregate"] = {
        "definition": "unweighted macro mean over domains; never used for selection",
        **{k: float(np.mean([m[k] for m in result["domains"].values()]))
           for k in scalar_keys if all(m.get(k) is not None for m in result["domains"].values())},
        "pooled_all_metrics": metrics({
            k: np.concatenate([domain[k] for domain in pooled]) for k in pooled[0]
        }),
        "pooled_definition": "sample-weighted concatenation of all dev domains; global "
                             "orientation uses only available labels; never used for selection",
    }
    with (out / "evaluations.jsonl").open("a") as f:
        f.write(json.dumps(result, allow_nan=False) + "\n")
    return result


def verify_freeze(config):
    root = ROOT / config["split_dir"]
    audit = json.loads((ROOT / config["audit_dir"] / "audit.json").read_text())
    assert audit["passed"]
    assert audit["config_sha256"] == base.sha256(ROOT / config["_config_path"])
    assert audit["freeze_sha256"] == base.sha256(root / "freeze.json")
    frozen = json.loads((root / "freeze.json").read_text())
    assert frozen["config_sha256"] == base.sha256(ROOT / config["_config_path"])
    for name, info in frozen["split_files"].items():
        assert base.sha256(root / name) == info["sha256"], name
    for name, digest in frozen["data_hashes"].items():
        assert base.sha256(ROOT / name) == digest, name
    # Validate every training/development image against the frozen inventory.
    # E4 gets manifest hash verification only.
    for split in ("train", "dev"):
        for line in (root / f"{split}.jsonl").open():
            row = json.loads(line)
            assert base.sha256(ROOT / row["image"]) == row["image_sha256"], row["image"]
            if "mask" in row:
                assert base.sha256(ROOT / row["mask"]) == row["mask_sha256"], row["mask"]
    for split in ("train", "dev"):
        root = ROOT / config["datasets"][f"dexycb_{split}"]
        manifest = json.loads((root / "sealed_manifest.json").read_text())
        for name, info in manifest["extracted_files"].items():
            assert base.sha256(root / name) == info["sha256"], name


def checkpoint(path, model, optimizer, step, config):
    temp = path.with_suffix(".tmp")
    base.save_checkpoint(temp, model, optimizer, step, config)
    temp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--freeze", action="store_true")
    p.add_argument("--run-id")
    p.add_argument("--device", default="cuda:1")
    a = p.parse_args()
    config = json.loads((ROOT / a.config).read_text())
    config["_config_path"] = a.config
    assert config["stage"] == "M-S1"
    assert not any(config["model"][k] for k in ("teacher", "knowledge_distillation", "adversarial_loss"))
    assert config["positive_camera"]["minimum_scale"] == .05
    assert all(config["scope"].values())
    assert not any("final_e4" in value or "subject-01" in value
                   for value in config["datasets"].values())
    if a.freeze:
        freeze(config)
        return
    verify_freeze(config)
    out = ROOT / "artifacts" / a.run_id
    out.mkdir(parents=True, exist_ok=False)
    for path in (ROOT / a.config, Path(__file__), Path(base.__file__)):
        shutil.copy2(path, out / path.name)
    shutil.copytree(ROOT / config["split_dir"], out / "splits")
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    device = torch.device(a.device)
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    data, dev = datasets(config, "train"), datasets(config, "dev")
    training = config["training"]
    loaders = {d: DataLoader(ds, batch_size=training["eval_batch_size"], shuffle=False,
                             num_workers=training["eval_workers"], pin_memory=True)
               for d, ds in dev.items()}
    diagnostic = {}
    # Train curves only: a deterministic, unaugmented panel, never checkpoint selection.
    for d, ds in data.items():
        import copy
        ds = copy.copy(ds)
        if isinstance(ds, Subset):
            ds.dataset = copy.copy(ds.dataset)
            ds.dataset.train = False
        else:
            ds.train = False
        diagnostic[d] = base.make_eval_loader(ds, 256, training["eval_batch_size"],
                                              training["eval_workers"])
    sampler = SequenceMixSampler(data, config)
    loader = DataLoader(ConcatDataset(list(data.values())), batch_sampler=sampler,
                        num_workers=training["workers"], pin_memory=True,
                        worker_init_fn=base.worker_init, persistent_workers=True)
    model, cfg, weights = base.configure_model(config)
    model.to(device)
    (out / "effective_config.yaml").write_text(cfg.dump())
    base.write_json(out / "provenance.json", {
        "config_sha256": base.sha256(ROOT / a.config),
        "script_sha256": base.sha256(Path(__file__)),
        "base_script_sha256": base.sha256(Path(base.__file__)),
        "imagenet_sha256": base.sha256(weights),
        "train_counts": {d: len(ds) for d, ds in data.items()},
        "dev_counts": {d: len(ds) for d, ds in dev.items()},
        "initialization": "Fresh ImageNet1K V2 backbone and new head; no smoke checkpoint",
    })
    # Baseline uses exactly the same crops, domains, masks and full independent dev.
    fresh_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    old = ROOT / config["baseline_checkpoint"]
    assert base.sha256(old) == config["baseline_sha256"]
    model.load_state_dict(torch.load(old, map_location="cpu")["state_dict"], strict=True)
    baseline = evaluate(model, loaders, device, out, "rhd_only_baseline")
    model.load_state_dict(fresh_state, strict=True)
    del fresh_state
    optimizer = torch.optim.AdamW(model.get_parameters(), lr=training["learning_rate"],
                                 weight_decay=training["weight_decay"],
                                 betas=tuple(training["betas"]), eps=training["eps"])
    initial = evaluate(model, loaders, device, out, "dev_0000000")
    evaluate(model, diagnostic, device, out, "train_diagnostic_0000000")
    best, best_step, stale = float("inf"), None, 0
    collapse_streak = {d: int(initial["domains"][d]["projection_collapse_rate"] > 0) for d in DOMAINS}
    any_collapse_streak = int(any(collapse_streak.values()))
    stop_reason, last_step = "budget_completed", 0
    exposure = Counter()
    start = time.monotonic()
    with (out / "steps.jsonl").open("a", buffering=1) as log:
        for step, batch in enumerate(loader, 1):
            exposure.update(batch["sample_id"])
            warmup = training["warmup_updates"]
            factor = step / warmup if step <= warmup else (
                training["minimum_lr_fraction"] + (1 - training["minimum_lr_fraction"]) *
                .5 * (1 + math.cos(math.pi * (step - warmup) / (training["updates"] - warmup))))
            for group in optimizer.param_groups:
                group["lr"] = training["learning_rate"] * factor
            moved = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            model.train()
            optimizer.zero_grad(set_to_none=True)
            pred = model.forward_step({"img": moved["img"]}, train=True)
            total, terms, counts, scalars = base.compute_losses(pred, moved, config["losses"])
            if not torch.isfinite(total):
                stop_reason = "nonfinite_training_loss"
                log.write(json.dumps({"step": step, "failure": stop_reason}) + "\n")
                break
            gradients = None
            if step == 1 or step % training["gradient_log_interval"] == 0:
                targets = [model.mano_head.decpose.weight, model.mano_head.decshape.weight,
                           model.mano_head.deccam.weight, next(model.backbone.parameters())]
                gradients = {}
                for name, term in terms.items():
                    grads = torch.autograd.grad(term * config["losses"][name], targets,
                                                retain_graph=True, allow_unused=True)
                    gradients[name] = [None if g is None else float(g.norm()) for g in grads]
            total.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.get_parameters(),
                                                  training["clip_grad_norm"])
            if not torch.isfinite(grad):
                stop_reason = "nonfinite_gradient"
                log.write(json.dumps({"step": step, "failure": stop_reason}) + "\n")
                break
            optimizer.step()
            last_step = step
            uv = pred["pred_keypoints_2d"].detach()
            span = (uv.amax(1) - uv.amin(1)).amax(1) * 256
            depth = base.camera_depth(pred).detach()
            camera = pred["pred_cam"].detach()
            row = {"step": step, "loss": float(total.detach()),
                   "components": {k: float(v.detach()) for k, v in terms.items()},
                   "effective_sample_counts": counts, "effective_scalar_counts": scalars,
                   "gradient_components_decpose_decshape_deccam_backbone": gradients,
                   "gradient_norm_before_clip": float(grad),
                   "projection_span_px_min_median_max": [float(span.min()), float(span.median()), float(span.max())],
                   "weak_camera_scale_min_median_max": [float(camera[:, 0].min()),
                       float(camera[:, 0].median()), float(camera[:, 0].max())],
                   "projection_collapse_rate": float((span < 5).float().mean()),
                   "negative_depth_rate": float((depth <= 0).any(1).float().mean()),
                   "lr": optimizer.param_groups[0]["lr"], "elapsed_s": time.monotonic() - start}
            log.write(json.dumps(row, allow_nan=False) + "\n")
            if step % 50 == 0:
                print(json.dumps(row), flush=True)
            if step % training["eval_interval"] == 0 or step == training["updates"]:
                result = evaluate(model, loaders, device, out, f"dev_{step:07d}")
                evaluate(model, diagnostic, device, out, f"train_diagnostic_{step:07d}")
                safe = True
                for d, m in result["domains"].items():
                    collapse_streak[d] = collapse_streak[d] + 1 if m["projection_collapse_rate"] > 0 else 0
                    safe &= m["failure_rate"] == 0
                any_collapse_streak = (
                    any_collapse_streak + 1 if any(
                        m["projection_collapse_rate"] > 0 for m in result["domains"].values()) else 0)
                regressions = []
                for d in ("freihand", "rhd"):
                    for metric in ("wrist_relative_mpjpe_mm", "fingertip_wrist_relative_error_mm",
                                   "visible_2d_epe_px"):
                        observed = result["domains"][d].get(metric)
                        reference = baseline["domains"][d].get(metric)
                        if observed is not None and reference is not None and observed > reference * 1.25:
                            regressions.append({"domain": d, "metric": metric,
                                                "observed": observed, "baseline": reference})
                if regressions:
                    event = {"step": step, "domain_imbalance": regressions,
                             "cumulative_domain_exposures": {
                                 d: sum(v for k, v in exposure.items() if k.startswith(
                                     {"dexycb": "20200813", "freihand": "freihand:", "rhd": "rhd-"}[d]))
                                 for d in DOMAINS},
                             "action": "retain frozen sampler for this registered run; "
                                       "diagnose and register a sampler correction if persistent"}
                    with (out / "domain_imbalance_events.jsonl").open("a") as events:
                        events.write(json.dumps(event) + "\n")
                    print(json.dumps(event), flush=True)
                value = result["domains"]["dexycb"].get("wrist_relative_mpjpe_mm", float("inf"))
                if safe and value < best - training["min_improvement_mm"]:
                    best, best_step, stale = value, step, 0
                    checkpoint(out / "best_dev.pt", model, optimizer, step, config)
                else:
                    stale += 1
                checkpoint(out / "last.pt", model, optimizer, step, config)
                state = {"step": step, "best_step": best_step, "best_mpjpe": best if best_step else None,
                         "stale_evaluations": stale, "collapse_streak": collapse_streak,
                         "any_domain_collapse_streak": any_collapse_streak}
                base.write_json(out / "selection_state.json", state)
                if any_collapse_streak >= 2:
                    stop_reason = "consecutive_projection_collapse"
                    break
                if any(m["nonfinite_rate"] > 0 for m in result["domains"].values()):
                    stop_reason = "nonfinite_dev"
                    break
                if stale >= training["patience_evaluations"]:
                    stop_reason = "dev_patience"
                    break
    checkpoint(out / "last.pt", model, optimizer, last_step, config)
    base.write_json(out / "training_sample_exposure.json", dict(exposure))
    hashes = {name: base.sha256(out / name) for name in ("last.pt", "best_dev.pt")
              if (out / name).exists()}
    reload_checks = {}
    reload_images = next(iter(loaders["dexycb"]))["img"][:4].to(device)
    model.eval()
    for name in hashes:
        state = torch.load(out / name, map_location="cpu")["state_dict"]
        model.load_state_dict(state, strict=True)
        with torch.no_grad():
            before = model.forward_step({"img": reload_images})["pred_keypoints_2d"].cpu()
        model.load_state_dict(state, strict=True)
        with torch.no_grad():
            after = model.forward_step({"img": reload_images})["pred_keypoints_2d"].cpu()
        error = float((before - after).abs().max())
        reload_checks[name] = {"strict_load": True,
                               "repeat_forward_max_error": error if math.isfinite(error) else None}
        if math.isfinite(error):
            assert error < 1e-7
    base.write_json(out / "completion.json", {
        "stage": "M-S1", "stop_reason": stop_reason, "last_step": last_step,
        "best_step": best_step, "checkpoint_hashes": hashes, "strict_reload": reload_checks,
        "forbidden_actions_executed": [], "m_s2_started": False,
    })
    print(json.dumps({"completed": stop_reason, "step": last_step, "hashes": hashes}), flush=True)


if __name__ == "__main__":
    main()
