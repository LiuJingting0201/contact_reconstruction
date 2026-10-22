# Dyn-HaMR experiment record — 2026-10-19

## Status and evidence boundary

**Latest, 2026-10-20:** The real-E2 pose-preserving ablation is complete. Stereo Huber translation-only (B2) is selected: **8.125 mm absolute MPJPE**, unchanged **6.762 mm wrist-relative MPJPE**, 1/32 quality failures and exact orientation/pose/shape freezing. It outperforms the prior translation+orientation backend while running in about half the time. All 352 ablation states and final metrics passed independent verification. The backend-selection gate is open for a future E3, but ResNet/E3 remain paused. See “Real E2 pose-preserving backend ablation” at the end.

**Historical status before real E2:** Geometry experiments, real-image HaMeR inference, a 20-step pseudo-label smoke, and two 200-step RHD ground-truth supervised pilots were complete. The RHD pilots use 128 training / 32 official evaluation images, without KD. They reveal projection collapse with default initialization and limited generalization after an initialization change; they are not a full M1 baseline. At that time calibrated real stereo fitting, an untouched final benchmark and deployment benchmarks had not run.

Update at 09:50 UTC: both supervised pilots completed with strict checkpoint reload difference 0; 121 remote evidence file hashes and all 20 split/step evaluation records were verified locally. The original missing-MANO blocker was resolved using the author's public HaMeR demo resources. The ledger below records the state at each time; newer sections supersede earlier pending items.

Execution protocol: [v1.1](dyn_hamr_reproduction_protocol.md). Original meeting and both original plans are preserved under [reference](reference/). Local archive copies were made before revision. CPU experiment parameters and pass criteria were written before execution and copied into the run directory.

## Remote environment

| Item | Observed value |
| --- | --- |
| SSH alias | `simready-dev` |
| Host | `di-20260827194854-gvzks` |
| Project | `/workspace/pose_reconstract` |
| GPUs | 2 × NVIDIA L20, 46068 MiB each |
| Driver | 535.154.05 |
| Initial usage | GPU 0: 3851 MiB / 10%; GPU 1: 291 MiB / 0% |
| Selected device | Physical GPU 1, `CUDA_VISIBLE_DEVICES=1` |
| Initial disk availability | 36 GB |
| Geometry environment | `.venv`, managed Python 3.10.21; NumPy 1.26.4, SciPy 1.13.1 |
| GPU environment | `.venv-gpu`, Python 3.10.13; PyTorch 2.2.0, torchvision 0.17.0, CUDA 12.1 |
| GPU dependency source | Explicit `--system-site-packages` inheritance from `/opt/simready/pamo-env` |

The GPU environment depends on the existing parent environment. It is not fully hermetic. New dependencies are installed in the project venv; the parent and Isaac Lab are unmodified. Only this task's slow download processes were stopped. No pre-existing GPU process was stopped.

## Operation ledger

Times below are UTC; add 8 hours for Beijing time. Full command arguments, stdout/stderr and return codes are under [local command logs](../artifacts/commands/) and [remote command logs](../artifacts/remote/commands/).

| Time | Operation | Observation / outcome |
| --- | --- | --- |
| 08:23 | Fetch selected public source files at fixed commits | 32 files fetched with URL, commit and SHA-256; partial audit snapshot, not full runnable repository |
| 08:24 | Create local venv and install geometry dependencies | Completed before user interruption; used subsequently for wheel preparation and result verification |
| 08:35 | Retrieve historical SSH alias and query developer machine | Connected; two L20 GPUs confirmed |
| 08:36–08:38 | Inventory established project paths and interpreters | uv exists at `/root/.local/bin/uv`; no relevant MANO/checkpoint/calibration found |
| 08:38 | Create remote Python 3.10 venv and download geometry dependencies | Interpreter installed; package download too slow |
| 08:42–08:43 | Download Linux wheels locally, transfer via rsync | Selected wheels successfully transferred; preserved slow-attempt logs |
| 08:43 | Stop this task's slow bootstrap; restart from local wheel directory | No experiment ran in the interrupted installation attempt |
| 08:44:04 | Run `cpu_geom_v1_001` | 41 cases; all 18 explicit checks passed |
| 08:44:07–09 | Create GPU venv; CUDA matmul and backward smoke test | Passed on L20; CPU/GPU float64 maximum difference 0 |
| 08:44:49 | Execute original camera loader definitions with controlled input | Reproduced intrinsics replacement and random extrinsics modification |
| 08:45–08:47 | Fetch pinned full repository archives | HaMeR and Fast-HaMeR complete; large Dyn-HaMR archive timed out and is incomplete |
| 08:48 onward | Install frontend import dependencies and smoke-test interfaces | Missing wheel builder and transitive dependencies diagnosed and repaired; every failed attempt retained |
| 08:52:20 | Final frontend import smoke and dependency consistency check | All 8 imports including `hamer.models` passed; `pip check`: no broken requirements |

The initial asset search used a broad substring and returned `humanoid_*.npz` motion files. These are **false positives**, not MANO assets. The search was corrected to the `mano` filename prefix.

## CPU geometry results

Run: [cpu_geom_v1_001](../artifacts/remote/geometry/cpu_geom_v1_001/). Executed on the developer machine, 08:44:04.376–08:44:04.448 UTC. Reported 0.0723 s is this small synthetic script's wall time, **not pipeline latency or a model benchmark**.

Input: 21 synthetic non-MANO points, seed 42, meters, float64. Reference K includes unequal focal lengths, an off-center principal point and skew. Stereo baseline 0.06 m; rectified focal length 1000 px. Shared-state fit optimizes only three translation parameters with fixed point geometry.

| Experiment | Measured result | Interpretation |
| --- | --- | --- |
| General-K linear translation fit | Error `2.24e-16 m`; condition 14.08 | Equation and unit consistency on this controlled input |
| Shared stereo translation fit | Error `1.13e-16 m`; 8 `nfev`, 29 residual calls | Known shape + ideal observations recover translation |
| Root offset transform | Maximum error `1.15e-16 m` | Synthetic rigid transform identity holds |
| Omit root offset | 3.876 mm discrepancy | Nonzero offsets matter; not a MANO integration test |
| Double crop reflection | Mean pixel error 256.84 px | Repeating a mirror operation can dominate all fitting gains |
| Depth 0.5 m, disparity σ≈1 px | Depth std 4.1688 mm; theory 4.1667 mm; P95 abs 8.1135 mm | Matches first-order uncertainty propagation |
| Depth 1.0 m, same noise | Depth std 16.7570 mm; theory 16.6667 mm; P95 abs 32.9445 mm | Depth uncertainty grows approximately quadratically |
| 4 right-view outliers, L2 | Translation error 24.9654 mm | Controlled corruption biases ordinary fitting |
| Same outliers, Huber | Translation error 0.8660 mm; ratio 0.03469 | Robust loss helps this fixture; not a general performance guarantee |
| 10% / 30% missing observations | Both noiseless checks passed | Remaining points suffice in this geometry |
| All missing / rank deficient / negative-depth solution | All explicitly rejected | Invalid outputs are not silently reported as valid |
| Partial NaN | 20 retained points; error `4.48e-16 m` | Invalid-point masking works |
| Scale shape and translation by 1.2 | Left residual `2.54e-13 px`, root discrepancy 120.08 mm; right residual 16.65 px | Low monocular residual cannot establish metric correctness |

Sensitivity cases use independent analytical rectified-stereo equations, not the MANO solver. A moving hand at 0.5 m/s and 30 fps with a one-frame right-camera offset produces:

| True depth | Right image one frame earlier | Right image one frame later |
| --- | ---: | ---: |
| 0.5 m | −108.70 mm depth error | +192.31 mm |
| 1.0 m | −217.39 mm | +384.62 mm |

Asymmetry follows the reciprocal disparity-to-depth relation. It is not an observed property of the user's camera. ±1% baseline error produces ±1% depth bias; ±1%/±5% focal error likewise produces those relative biases in this controlled equation.

### Unexpected magnitude in the focal fixture

The deliberately fixed weak-camera scale `s=1`, bbox=220 px and 1920×1080 image give virtual full-image focal lengths 3750/37500 px and decoded depths 34.09/340.91 m. Translation fitting with real K yields about 9.09 m, despite only 0.166/0.220 px mean fitting residual.

These unrealistic distances are **not a real model prediction**. The arbitrary weak-camera scale is not calibrated to the random point cloud's image extent. The result illustrates why correct focal units and low residual alone do not establish a plausible hand distance. This fixture cannot validate the meeting's claim that the real-focal branch has accurate depth.

## Original camera loader: confirmed defects

Evidence: [camera loader probe](../artifacts/remote/camera_loader_probe/). The script executes the exact upstream `CameraData` and `load_cameras_npz` AST definitions, with only the logger supplied externally. This is an isolated component experiment, not a full pipeline run.

Source SHA-256: `ce8ea3f399c1dbda5d5139c237da4b414837babb8401b326507fe733a6dd4f8d`.

Input: two identical cameras, `[fx,fy,cx,cy]=[1000,930,931,517]`, `R=I`, `t=[0.03,-0.02,0.01] m`.

| Setting | Intrinsics outcome | Translation outcome |
| --- | --- | --- |
| `is_static=True`, seed 42 | Replaced with `[1500,1500,960,540]`; max difference 570 px | Replaced by zero |
| `is_static=False`, seed 42 | Preserved | 35.713 mm difference from input |
| `is_static=False`, seed 43 | Preserved | 120.434 mm difference from input |

The static branch also prints that the camera file “does not exist” even when it exists. The nonstatic branch adds an unrecorded random translation through `torch.randn(3)*0.1`. A calibrated adapter must preserve both K and extrinsics, with explicit tests. Neither original branch is approved for the real calibrated-stereo experiment.

## Reproducibility and known limitations

- Saved inputs and outputs: [raw arrays](../artifacts/remote/geometry/cpu_geom_v1_001/raw_arrays.npz), [all metrics](../artifacts/remote/geometry/cpu_geom_v1_001/metrics.csv), [every solver call](../artifacts/remote/geometry/cpu_geom_v1_001/solver_traces.json), protocol and script snapshots.
- [Verification](../artifacts/verification.json) checks all geometry artifact SHA-256 values and independently recomputes the two reported depth standard deviations from the saved arrays.
- Expected fault injections are reported separately from normal measurements; failed setup commands are retained rather than removed.
- The CPU stage did not test crop interpolation, handedness labels, MANO joint conventions or differentiable MANO gradients. Subsequent RHD runs exercise crops, the documented joint mapping and MANO gradients; real lens undistortion and timestamp alignment remain untested.
- SciPy translation fitting is not the original Dyn-HaMR LBFGS optimizer. No conclusions about its MANO closure count or real throughput follow.
- E0 video baseline, full G0–G4/V1–V5 matrices, full-scale M1/M2 supervised training, YOLO training and TensorRT remain unexecuted. The later RHD pilot is a small public-data training experiment.
- At 08:52 only imports had passed. The later phase below loads real weights, instantiates MANO and executes training. `detectron2`, legacy `mmcv`, SLAM and full upstream training dependencies remain outside this minimal regressor environment.

## Inputs needed for the next real-model experiment

The initial 6,037,554,929-byte archive was reachable but too slow. It was superseded by direct file downloads from the author's public Space; the partial archive is retained as a failed-transfer artifact. Local searches did not find MANO, but the later official download did.

For real stereo validation, two synchronized views plus K, distortion, extrinsics and timestamps are still required. A real 3D improvement claim additionally requires independent references. Public supervised data has now been obtained for the RHD pilot below.

## Real model execution and training — 09:17–09:18 UTC

Protocol and parameters were added before execution in section 17 and `experiments/real_image_smoke.json`.

### Official assets and transfer

Source: author's Hugging Face Space `geopavlakos/HaMeR`, revision `aabc78ee4fd1f1ccf1a55fdd4225bf84df0e680a`. The public Space includes MANO, mean parameters, model configuration and the checkpoint. Asset paths, URLs and hashes: [manifest](../artifacts/official_asset_manifest.json).

Checkpoint: 2,689,536,166 bytes, SHA-256 `e5cc06f294d88a92dee24e603480aab04de532b49f0e08200804ee7d90e16f53`, matched the author's LFS metadata locally and after transfer. The slower 6 GB archive download was stopped; only this task's process was terminated. The first transfer command used an unsupported local rsync option; it failed before copying. The corrected transfer completed in 261.7 s.

### Real-image focal comparison

Input: pinned HaMeR sample `test1.jpg`, manually selected visible right hand, bbox `[253,151,388,342]`, rescale 2.0. No manual keypoint ground truth or real calibration was provided. [Outputs and overlays](../artifacts/remote/real_image_smoke_v1/).

| Measurement | Result |
| --- | --- |
| Checkpoint missing / unexpected keys | 0 / 0 |
| Focal cfg 500, crop camera z | 0.703576 |
| Focal cfg 5000, crop camera z | 7.035760 |
| Maximum difference in MANO parameters | 0 |
| Maximum difference in weak camera | 0 |
| Mean full-image projected-joint difference | 0.86945 px |
| Nonfinite outputs | None |
| Peak allocated GPU memory | 2,782,339,584 bytes |

This experimentally verifies that changing postprocessing focal length leaves the network's MANO and weak camera outputs unchanged on this sample. Both overlays were saved; the 5000 overlay visually places joints on the selected hand, but visual inspection is not an independent accuracy metric. Neither camera z value is a calibrated depth estimate. The first forward took 0.795 s and the second 0.0207 s; cold-start effects make these values unsuitable for a speedup claim.

### ResNet/MANO training smoke

[Run directory](../artifacts/remote/resnet_pseudolabel_smoke_v1/). ResNet-50 ImageNet V2 backbone, original Fast-HaMeR MANO decoder, all backbone/head parameters trainable. Twenty FP32 AdamW updates on two copies of the same real crop; frozen 5000-branch HaMeR outputs provide pseudo targets.

| Measurement | Result |
| --- | --- |
| Actual optimizer updates | 20 |
| Initial training loss | 0.01885320 |
| Last training loss | 0.0005911613 |
| Maximum change in first backbone parameter | 0.000160158 |
| Nonfinite loss/gradient | None |
| Strict checkpoint reload joint difference | 0 m |
| Peak allocated GPU memory | 1,424,951,808 bytes |
| Training + save/reload wall time | 4.2405 s |

This proves the model can train, update parameters and reload on this GPU environment. It does not establish useful reconstruction generalization: there is only one unique image, no validation set, and labels come from a teacher. It must not be counted as M1 supervised no-KD performance. Per-step loss components, gradient norms and times are retained; the remote checkpoint contains model and optimizer state.

### Calibrated adapter

`scripts/calibrated_camera.py` preserves supplied K/R/t without random recentering and rejects invalid calibration. [Eight component tests](../artifacts/remote/calibrated_camera_v1/results.json) passed, including seed independence, stereo baseline sign, NaN, reflection, missing K, resolution and frame-count mismatches. This adapter has not yet been integrated into the full original pipeline.

## RHD supervised training pilots — 09:33–09:46 UTC

Protocol: sections 18–18.3. Configurations: [v1](../experiments/rhd_supervised_v1.json), [v2](../experiments/rhd_supervised_v2.json), [v3](../experiments/rhd_supervised_v3.json). These are **synthetic rendered** RGB data with ground-truth 2D/3D joints, not real camera recordings. No teacher targets or smoke checkpoint were used.

### Dataset and annotation audit

Source: Freiburg's official `RHD_v1-1.zip`, 7,080,690,299 bytes. The ZIP directory was read remotely, then selected annotations/images were fetched by byte range, decompressed and checked against ZIP CRC32 and lengths. Local SHA-256 values verify transfers; no independent whole-archive hash was provided. Preserve the dataset's research-only, noncommercial README and cite Zimmermann & Brox (ICCV 2017).

Selection is deterministic (seed 42): 128 images from 26,305 eligible official training samples, 32 from 1,556 eligible official evaluation samples. Only right hands, >=16 visible joints, all joints inside 320x320, positive z, bbox extent >=40 px. These filters bias the pilot toward visible hands. The source manifest records every ID, image hash, annotation hash and label hash.

Right-hand indices 21..41 are reordered from tip-to-palm into HaMeR's base-to-tip order. Official example code identifies XYZ units as meters. GT square crops use bbox expansion 2 and 256x256 output, ImageNet normalization, no mirror/augmentation. Affine inversion error is checked below 0.001 px for every selected crop. First-eight GT overlays from each split were inspected; the visible joints align with the rendered hands. This visual check does not prove anatomical equivalence of RHD joints and MANO.

| Attempt / time | Observation | Handling |
| --- | --- | --- |
| v1 extraction, 09:33–09:36 | Image 302 XYZ/K projection differs from UV by 0.05061 px; fails preregistered 0.01 px threshold | Preserve failed command and v1 config; zero training updates |
| Full annotation audit, 09:37–09:38 | Median all-joint discrepancies ~0.041 px; unfiltered data also include behind-camera / near-zero-depth points with extreme errors | Separate unfiltered diagnostics from selected visible-hand statistics |
| Selected-data float64 audit | Maximum 0.097030 px train / 0.082302 px eval; medians 0.037059 / 0.039407 px | Evidence consistent with finite annotation precision; exact upstream cause unproven |
| v2 extraction, 09:39 | Explicit new 0.25 px source-consistency tolerance; same selected IDs/labels | Pass; original v1 remains failed |
| v2 first model attempt, 09:40 | Initial predictions complete, OpenCV rejects noncontiguous overlay array | Preserve partial run; convert preview array to contiguous layout; zero training updates |
| v2 run02, 09:41–09:42 | 200 updates, finite gradients and exact reload | Completed; projection collapse discovered in saved outputs |
| v3, 09:45–09:46 | 200 updates with explicit stable head/camera initialization | Completed; collapse reduced but 3D generalization not improved |

Audit evidence: [all annotations](../artifacts/rhd_annotation_audit.json), [selected subset](../artifacts/rhd_annotation_audit_filtered.json). Raw source data and [selection manifest](../inputs/rhd/manifest.json) are retained locally and on the developer machine. Tool-session handles sometimes expired and the UI once reported a sandbox-process error after a transfer; persisted command records show the data rsync exited 0, and file hashes were checked by the model loader before execution.

### Training settings and complete metric trajectory

Both runs: physical GPU 1; NEW ImageNet V2 ResNet-50 + Fast-HaMeR MANO head, FP32, seed 42, AdamW LR 1e-5, weight decay 1e-4, batch 8, 200 updates (1,600 sample presentations), gradient clip 1. No augmentation, KD, adversarial loss or MANO rotation labels. BatchNorm running statistics are frozen; affine weights remain trainable.

Losses: wrist-relative 3D elementwise MSE in meters ×1, visible normalized-crop 2D elementwise MSE ×0.01, betas squared prior ×0.000005. MPJPE includes all 21 joints, including the zero-error wrist; no scale/rotation alignment. 2D metrics are visible-joint Euclidean error in 256x256 crop pixels. These are not absolute root-depth metrics.

v2 uses default upstream head initialization. v3 changes the combined readout initialization: Xavier gain 0.01, zero biases for pose/shape/camera readouts, weak-camera initial buffer `[8,0,0]`. This is an approximate hand-scale prior and does not use camera ground truth. Both runs use the same data order and final step 200; no earlier checkpoint was selected by evaluation performance.

| Run | Step | Train MPJPE mm | Evaluation MPJPE mm | Train 2D px | Evaluation 2D px |
| --- | ---: | ---: | ---: | ---: | ---: |
| v2 run02 | 0 | 156.522 | 153.937 | 520.227 | 524.702 |
| v2 run02 | 50 | 88.470 | 91.417 | 43.485 | 42.190 |
| v2 run02 | 100 | 67.903 | 81.486 | 42.665 | 41.714 |
| v2 run02 | 150 | 41.687 | 74.148 | 41.889 | 41.804 |
| v2 run02 | 200 | 37.351 | 75.825 | 41.442 | 41.669 |
| v3 | 0 | 165.682 | 156.410 | 80.132 | 79.140 |
| v3 | 50 | 87.106 | 91.576 | 39.114 | 41.988 |
| v3 | 100 | 54.772 | 76.895 | 31.419 | 38.667 |
| v3 | 150 | 41.831 | 77.240 | 27.866 | 38.313 |
| v3 | 200 | 36.089 | 77.024 | 25.734 | 39.009 |

### Abnormal behavior and interpretation

1. **v2 projection collapse:** final evaluation prediction bbox span is 0.015–6.295 px, median 2.025 px; GT span is 128 px. In the overlays, predicted points nearly coincide. A reduction from 524.70 to 41.67 px does not establish a useful reconstruction. The 3D point extent remains 0.102–0.163 m, so the entire 3D shape has not collapsed; the degeneracy is in projected extent. Initialization/camera optimization is a plausible contributor, but v2 did not save weak-camera arrays at each step, so its exact scale trajectory is not established.
2. **v3 reduces collapse:** final projected span is 64.889–131.693 px, median 84.639 px; weak-camera scale is 5.674–6.587, median 6.115. Evaluation 2D error is 2.660 px lower than v2, while MPJPE is 1.198 mm higher. This combined initialization change helps projection extent but does not demonstrate a 3D accuracy gain. The individual changes were not separately ablated.
3. **Limited generalization:** v2 evaluation MPJPE worsens from 74.148 to 75.825 mm between steps 150 and 200 while training improves; v3 training falls to 36.089 mm but evaluation remains 77.024 mm. This is consistent with overfitting / limited sample coverage. It is not proof that more training alone will improve accuracy.
4. **Evaluation reuse:** v2's 32 images did not contribute gradients. They were then used to diagnose v2 and design v3, so the v2/v3 comparison is exploratory development evidence. A new untouched test split is required for a final model-selection claim.

### Completion checks and saved evidence

| Check | v2 run02 | v3 |
| --- | ---: | ---: |
| Actual optimizer updates | 200 | 200 |
| Maximum first-optimizer-parameter update (Transformer positional embedding; corrected after runtime inspection) | 0.0002233982 | 0.0004507303 |
| Strict checkpoint reload maximum joint difference | 0 m | 0 m |
| Peak PyTorch allocated bytes | 2,071,006,208 | 2,071,006,208 |
| Timed train/evaluation/save/reload segment | 13.829 s | 13.847 s |

These timing segments start after model construction and GPU data preparation. They exclude download, crops/loading, setup and artifact transfer, and must not be advertised as end-to-end FPS.

Run directories: [v2 run02](../artifacts/remote/rhd_supervised_v2_run02/) and [v3](../artifacts/remote/rhd_supervised_v3/). Each contains `last.pt` (model + AdamW + RNG state), source/protocol/config snapshots, per-step sample IDs/loss/gradient/time, all five evaluations with per-image metrics, raw NPZ predictions/targets, GT/prediction overlays and training curves. Partial failures remain in their original directories. Local [verification](../artifacts/verification.json) checks remote artifact hashes and independently recomputes reported metrics from saved arrays.

The first post-v2 hash verification failed only on `rhd_pilot_run02_launcher.log`: the finalizer had hashed the log before its own completion output was appended. This is an evidence-finalization ordering defect, not evidence of a changed checkpoint. The failure is retained. The run scripts now finish before an external finalization command hashes their closed logs; no file is excluded to hide the mismatch. Source/config snapshots and identical v2/v3 batch order are also checked.

Final verification completed at 09:50 UTC: **121 remote evidence files match SHA-256**, both sets of 200 update records pass finite-loss/gradient and weighted-loss recomputation, all 20 split/step evaluations match raw arrays, source/config/manifest snapshot hashes match, and both runs have identical per-step sample IDs. Both final checkpoints are present locally. Verification result: `artifacts/verification.json`; full command record: `artifacts/commands/20261019T094958.813386Z/`.

The pilot establishes executable supervised training and identifies a concrete projection degeneracy. It does **not** establish a useful small-model replacement, full M1/M2 reproduction, KD benefit, stereo accuracy, or deployment throughput. Further accuracy work requires a larger diverse training split and a separate final test; calibrated real stereo remains a distinct data requirement.

## Post-hoc localization of remaining errors — 09:57–09:58 UTC

User follow-up: identify where the failure comes from. Entry: `scripts/diagnose_rhd_outputs.py`; raw results: [output diagnosis](../artifacts/rhd_output_diagnosis.json). Uses only previously saved targets/predictions, with no model changes or additional training.

| Diagnostic | v2 run02 | v3 | Meaning |
| --- | ---: | ---: | --- |
| Original evaluation MPJPE mm | 75.825 | 77.024 | Model output |
| One common wrist rotation fitted on training, then applied to evaluation | 74.967 | 76.573 | A single global rotation correction explains little |
| Per-image GT-oracle wrist rotation, no scale/translation/reflection, MPJPE mm | 22.845 | 21.937 | Much of the error is removable by changing whole-hand orientation |
| GT-oracle positive weak-camera fit, visible 2D error px | 31.876 | 30.641 | Camera refitting alone leaves substantial pose/projection error |
| Constant training-mean 3D pose baseline, MPJPE mm | 90.403 | 90.403 | Both models learn beyond a constant mean pose |

The common rotations fitted on training are only 3.313° / 2.534° and barely change evaluation error. This is evidence against a simple shared camera-axis rotation being the principal explanation. Per-image rotation alignment strongly suggests inaccurate instance-specific whole-hand orientation, though shape errors can also influence the fitted rotation; it does not prove a bug in the MANO global-orientation implementation. Oracle fitting uses evaluation ground truth and must never replace reported prediction accuracy.

The camera oracle minimizes squared normalized 2D residuals for a positive scalar and offset; it is a weak-perspective approximation and not a strict lower bound for mean Euclidean error under the perspective camera. The v3 final 50 steps have weighted mean 3D / 2D / shape losses `8.032e-4 / 7.447e-5 / 2.132e-6`. These scalar magnitudes alone do not establish gradient imbalance; component gradient measurement or a controlled loss-weight comparison is required.

Confirmed limitations in this pilot's training recipe: only 128 unique images, 200 steps (12.5 passes), no augmentation, newly initialized MANO head, indirect orientation supervision through joints, and camera supervision only through reprojection. Data coverage, initialization, pose supervision and objective balance therefore need controlled follow-up. The evidence does not establish that ResNet, RHD or the dual-focal/stereo method itself is defective.

## Runtime debugging and continued experiments — 2026-10-19, 10:00–10:20 UTC

User request: debug, fix and continue experiments against the meeting transcript. Protocol §19 and §19.1 were written before their corresponding runs. Physical GPU 1 only. Saved evidence includes full command stdout/stderr/return codes, HTTP debug events, source/protocol snapshots, per-step sample IDs/losses/gradients, predictions, overlays and model/optimizer checkpoints. No KD was introduced.

### Confirmed implementation defects and corrections

- **Invalid weak-camera geometry:** instrumented v2 reaches negative scales (-0.682 to -0.378) and negative crop-camera depths (-103.34 to -57.30 m) at step 10; step 100 includes virtual depths around 5,000 m. Finite-only checks missed this. The corrected runner applies `softplus(raw_scale)+0.05` before upstream projection, initializes the raw buffer by inverse softplus, checks positive joint depths/scales in training and evaluation, and uses the same transform on checkpoint reload. This enforces valid geometry; it does not guarantee accurate hand orientation or prevent every form of projection shrinkage.
- **Incorrect evidence label:** the first optimizer tensor is `mano_head.transformer.pos_embedding`, not the backbone. The earlier table is corrected. New runs separately measure actual ResNet `0.weight` changes; every post-fix run has a nonzero update and exactly matching saved/reloaded 2D and 3D output.
- **Missing deterministic training controls:** the unchanged instrumented v2/v3 reproduce step-0 predictions exactly but diverge after optimization. The attempted 1e-6 exact comparison fails; that failure remains recorded. Final evaluation v2 changes from 75.825/41.669 to 74.535 mm/41.918 px; v3 changes from 77.024/39.009 to 75.884 mm/37.088 px. Additional gradient evaluations do not intentionally alter parameters, but the original CUDA settings did not guarantee deterministic execution. Strict deterministic PyTorch/cuDNN/cuBLAS controls produce **zero difference** across two complete v4 runs at all five checkpoints, both splits, all saved 2D/3D/camera arrays. This resolves repeatability in the tested environment; it does not identify a particular CUDA kernel or promise cross-platform equality.

The default runner now uses corrected v4 plus deterministic execution. Historical non-deterministic configurations can be requested explicitly with `--no-deterministic`; saved run snapshots preserve their actual original code/settings. Increasing the 2D weight is retained as an experiment, not promoted as a confirmed fix.

### Coordinate and gradient controls

Direct MANO synthetic self-fit (four examples, 600 Adam updates) finishes at 0.020–0.099 mm MPJPE. Direct fitting to eight training RHD targets (1,000 updates) reaches 2.719–17.520 mm, mean 6.024 mm. This argues against a gross non-differentiable or globally incorrect MANO rotation/joint chain. It does not prove identical anatomical joint definitions; fitting uses GT and unconstrained betas and can encounter local minima. Arrays and all fit steps: `artifacts/remote/mano_debug_controls/`.

Observed pose-readout gradient norms confirm an objective imbalance in the original recipe: v2 step 200 weighted 3D/2D norm ratio is about 359; v3 about 8.89. Their alignment varies by step (v2 initial cosine -0.832; v3 final +0.839), so the terms are not always conflicting. Scalar loss magnitudes alone were insufficient evidence.

### Controlled post-fix training results

All metrics below use fixed final checkpoints; 3D is unaligned wrist-relative MPJPE over 21 joints, including the zero-error wrist. 2D is pooled visible-joint Euclidean error in the 256-pixel crop. All 200-step comparisons have identical sample order. The official 32-image evaluation subset is now a **reused development set**, not an untouched benchmark.

| Run | Steps | Train 3D mm | Development 3D mm | Train 2D px | Development 2D px |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 historical stable initialization | 200 | 36.089 | 77.024 | 25.734 | 39.009 |
| v4 positive camera | 200 | 36.335 | 77.208 | 26.196 | 39.101 |
| v5 positive camera, 2D weight 0.1 | 200 | 42.514 | 74.526 | 25.669 | 36.542 |
| v5 longer, fresh initialization | 1,000 | 17.684 | 78.472 | 14.286 | 40.405 |
| v4 deterministic A | 200 | 35.906 | 76.043 | 26.219 | 37.632 |
| v4 deterministic B | 200 | 35.906 | 76.043 | 26.219 | 37.632 |
| v5 deterministic | 200 | 40.525 | 75.323 | 27.125 | 38.812 |

Interpretation: the camera fix is a validity correction, with no established accuracy gain over already stabilized v3. The apparent 2D gain from reweighting in the first comparison **does not persist** in the deterministic comparison: 38.812 vs 37.632 px. A stronger 2D weight alone is not a reliable solution. Longer training substantially improves training accuracy while worsening development accuracy, supporting limited generalization/overfitting on 128 unique images. The next small-model accuracy experiment should expand diverse training data and use an untouched test split; simply adding more steps to this subset is not supported.

### Meeting dual-focal claim measured on the same development images

Entry: `scripts/teacher_geometry_control.py`; successful artifacts: `artifacts/remote/teacher_geometry_control_run02/`. Official pretrained HaMeR, identical square GT crops/normalization, internal ViT aspect crop retained. Changing focal configuration alone leaves MANO parameters and weak camera unchanged within 1e-6 for all 32 examples. “Two branches” here means two camera interpretations of the same predicted hand, not independently predicted hand shapes.

The nominal-5000 teacher has **7.185 px pooled visible 2D error** and **22.465 mm wrist-relative MPJPE**. It is much better than the small pilot but is not “absolutely accurate.” Pretraining overlap with RHD is unknown; this is a diagnostic reference, not a fair same-data model benchmark.

Use actual RHD K and crop scale to form the real-focal camera, apply crop-center/principal-point correction, and optimize translation only against the nominal pseudo 2D using robust least squares. Every fitted joint remains in front of the camera. All 32 pseudo-target and 32 oracle fits report successful termination.

| Camera interpretation / fitting target | Visible crop 2D error px | Absolute wrist-depth error mm |
| --- | ---: | ---: |
| Nominal-5000 pseudo projection | 7.197 | 4,523.913 |
| Real-K initialization | 18.495 | 57.817 |
| Real-K translation fitted to pseudo 2D | 7.195 | 64.442 |
| Real-K translation fitted to GT 2D (oracle) | 6.546 | 53.461 |

This table averages per-image visible errors, hence nominal 7.197 differs slightly from pooled 7.185. Depth includes MANO wrist offset plus translation; it is not translation-z alone. GT oracle is diagnostic and not deployable. The full-image pseudo residual falls from 8.351 to 2.662 px.

The observations support the meeting's qualitative projection/depth tradeoff and the usefulness of pseudo-2D reprojection. They do **not** validate the literal claims that nominal 2D is exact or real-focal depth is correct. Pseudo fitting improves 2D while mean depth error worsens by 6.625 mm. Monocular shape/scale/depth ambiguity remains. This experiment optimizes translation only; joint optimization over MANO pose/shape/root and calibrated stereo remain untested here. There is no synchronized real stereo dataset in this run, so no stereo accuracy claim is made.

### Failures and verification

The first instrumentation launch failed on a missing dictionary brace before model execution; its launcher log is retained. The first teacher attempt failed after the first forward because the original HaMeR branch returns MANO parameters as NumPy arrays, unlike Fast-HaMeR tensors. `run02` accepts both representations; the original partial directory and traceback remain. The first exact pre-reproduction assertion failed after step 0 and was not converted to a pass: summary output explicitly marks both original comparisons `all_arrays_match_1e_6: false`. Later deterministic repeats satisfy the original tolerance with exactly zero error.

`scripts/summarize_debug_experiments.py` independently recomputes every reported train/development evaluation from NPZ, checks all update sequences/weighted losses, positive geometry in corrected runs, identical first-200 batch IDs, actual backbone changes, reload equivalence, exact deterministic repeats, unchanged teacher MANO outputs across focal values, and real-K projection/depth metrics from saved joints/translations/K. Summary: `artifacts/debug_experiment_summary.json`; figure: `artifacts/remote/debug_comparison.png`. HTTP event log: `.dbg/trae-debug-log-hand-camera-collapse.ndjson`. Instrumentation and raw failures are intentionally retained.

Final transfer verification completed on 2026-10-20 at 03:09 UTC: all **365 remote evidence files** are present locally and match the remote SHA-256 manifest, including all 11 accumulated model/optimizer checkpoints. No partial checkpoint transfers remain. `artifacts/verification.json` reports zero hash errors; verification command and complete output are preserved in `artifacts/commands/20261020T030927.068367Z/`. This closes artifact synchronization; it does not change the accuracy limitations above.

## E1: full monocular MANO ablation — 2026-10-20

Following the requested E1 → E2 → E3 sequence, ResNet training remains paused. Protocol §20 was written before execution. Successful run: `artifacts/remote/e1_mano_run02/`, physical GPU 1, 32 cached pretrained HaMeR predictions, 800 Adam updates per variant. Total experiment runtime was 38.32 s, excluding earlier teacher inference, transfer and verification. This is an offline backend timing, not end-to-end latency.

MANO reconstruction from cached rotations/betas matches the teacher joints with maximum difference **2.98e-8 m**. Nominal/real branch parameters, IDs, and identical optimization initializations were checked. The objective fits all 21 nominal pseudo points using real K. GT visibility, UV and XYZ are evaluation-only, although the inherited crops use GT boxes. All metrics below are means of per-image metrics; full pose includes global orientation and 15 articulated rotations.

| Variables | Visible GT2D crop px | Root-relative MPJPE mm | Absolute MPJPE mm | Wrist depth error mm | Quality failures / 32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial real-K translation | 18.495 | 22.465 | 59.142 | 57.817 | 12 |
| Translation | 7.211 | 22.465 | 52.590 | 53.170 | 4 |
| Translation + global orientation | 7.158 | 23.282 | 45.859 | 47.690 | 5 |
| Translation + full pose | 7.301 | 23.866 | 49.094 | 48.654 | 5 |
| Translation + full pose + shape | 7.237 | 23.552 | 50.360 | 50.414 | 3 |

All four variants have **0/32 numerical failures**. The separately preregistered quality flag is visible crop 2D >20 px OR root-relative MPJPE >50 mm OR wrist-depth error >100 mm. Thus the full-pose+shape run has fewer threshold failures despite worse mean depth than the orientation-only run; a single average does not capture every sample. All finite samples, including quality failures, remain in the averages.

| Variables | Global rotation change deg | Mean articulated rotation change deg | Beta L2 change | Translation change mm |
| --- | ---: | ---: | ---: | ---: |
| Translation | 0 (frozen) | 0 (frozen) | 0 | 64.600 |
| Translation + global orientation | 10.510 | 0 (frozen) | 0 | 54.894 |
| Translation + full pose | 9.804 | 7.842 | 0 | 23.282 |
| Translation + full pose + shape | 9.734 | 7.046 | 1.225 | 16.185 |

The raw arccos-based angle metric has <0.01-degree floating-point noise even for unchanged rotations; frozen-array equality is checked exactly and shown as zero here. Final mean objectives are still decreasing over the last 100 updates by approximately 0.00617 / 0.00146 / 0.000337 / 0.000214 respectively. These are fixed-budget endpoints, not certified converged optima.

**Interpretation:** translation explains most of the 2D improvement. Adding global orientation gives the lowest mean absolute MPJPE and depth error in this run. Opening articulated pose and shape does not further improve these means and slightly worsens root-relative MPJPE compared with the teacher. It substantially reduces pseudo fitting residual without establishing a corresponding GT3D benefit. This supports testing the calibrated stereo constraint next; it does not establish which variables are universally optimal or validate small-model compensation.

This translation result differs from the earlier §19 robust SciPy fit (64.442 mm depth). E1 uses all pseudo points, including GT-occluded joints, L2 rather than soft-L1, and a fixed Adam budget; the earlier fit used GT visibility. Do not attribute the cross-run difference to a single factor or present it as identical reproduction.

Evidence: every sample's 801 states of rotations/betas/translation/objective, every step's loss components and gradient norms, final geometry, independent GT arrays and optimizer states are saved. `scripts/verify_mano_backend.py` independently recomputes real-K projections, per-image/aggregate metrics and failure flags, checks finite trajectories, matching initializations and frozen-parameter invariants; **passed for initial plus all four variants**. Figure: [E1 comparison](../artifacts/remote/e1_backend_comparison.png).

Failure retained: E1 run01 was terminated after discovering that this HaMeR checkout uses two consecutive 3D columns for rotation-6D, rather than interleaved 3x2 flattening. Run02 corrects the inverse mapping and requires a <1e-6 rotation roundtrip check. Run01 remains partial/invalid and is not used above.

E2 status: real synchronized stereo cannot start from the current project inputs, which contain only monocular RHD and model assets. Required inputs are paired timestamp-matched images, matching K and distortion handling, meter-scale world-to-camera extrinsics, and independent evaluation GT. The separate synthetic MANO backend control below cannot substitute for real E2. E3 remains gated on completion of real E2.

## E2 synthetic backend control — completed, not real-image E2

Run `artifacts/remote/e2_synthetic_run02/` completed six 800-update variants in 64.13 s on GPU 1. Eight generated MANO hands, 0.06 m calibrated baseline, nonidentity right rotation, 1 px observation noise, and four right-view joints corrupted by `[50,-35] px`. Identical initial poses/translations, with betas held at their known generating values. Confidence weighting uses an **oracle mask of the injected outliers**, not a learned occlusion estimator. These favorable conditions are not a model-accuracy benchmark.

| Variant | Independent GT2D px | Absolute MPJPE mm | Wrist depth error mm | Numerical / quality failures |
| --- | ---: | ---: | ---: | ---: |
| Initial (both views) | 24.795 | 62.480 | 60.000 | 0 / 6 |
| Left-only | 1.144 | 8.320 | 4.620 | 0 / 0 |
| Right-only, corrupted | 12.583 | 35.141 | 24.753 | 0 / 0 |
| Stereo L2, corrupted | 7.731 | 39.124 | 28.927 | 0 / 0 |
| Stereo Huber, corrupted | 4.584 | 10.985 | 3.293 | 0 / 0 |
| Stereo + oracle confidence | 0.913 | 5.083 | 2.241 | 0 / 0 |
| Clean noiseless stereo control | 0.071 | 0.903 | 0.387 | 0 / 0 |

GT2D is evaluated in each variant's fitted view(s), against uncorrupted generating projections; left/right single-view values therefore do not summarize held-out-view error. XYZ and wrist depth are always evaluated in the shared left-camera reference frame, including the right-only solve. Each row has eight samples. The quality thresholds are the same diagnostic thresholds as E1, with full pixels used because this synthetic fixture has no crop; zero threshold failures does not mean equivalent accuracy.

The clean control reaches mean absolute MPJPE 0.903 mm, supporting the shared geometry and arbitrary-camera transform implementation. With severe right-view outliers, plain stereo becomes worse than the clean left-only fit. Huber improves wrist depth considerably but absolute MPJPE remains worse than left-only; correct masking gives the best noisy-control metrics. Stereo is therefore not automatically beneficial when one view has bad pseudo labels. This is a controlled solver observation, not validation of the meeting's real-image stereo claim or of E3 compensation.

The independent verifier passed initial plus all six variants, checking projections, metrics/failure flags, 801 saved states per variant, equal initialization, fixed betas and finite trajectories. First synthetic attempt failed its rotation roundtrip assertion before optimization because the legacy mean-pose array gave poorly conditioned columns under this checkout's decoder. Run02 uses explicit identity columns plus seeded perturbations for fixture generation, preserving the assertion and all original failure files.

Final synchronization at 05:41 UTC: **446 remote evidence files verified by SHA-256, zero mismatches**, including successful/failed E1/E2 attempts, all parameter trajectories, optimizer states, raw command logs and the comparison figure. Successful E1 and synthetic-E2 wrappers both returned code 0 and closed before manifest generation. E1 archives occupy about 42 MB and synthetic-E2 about 28 MB locally. Verification output: `artifacts/commands/20261020T054101.684415Z/`; manifest: `artifacts/remote/remote_sha256.json`. No new ResNet training ran.

## Real E2 — input blocked; executable interface delivered (2026-10-20)

### 1. Real E2 data and calibration

**真实 E2 未执行，因此会议的真实双目收益尚未验证。**

Inherited protocol, full E1/synthetic-E2 code/configs/results, artifact verification,
and available source changes were inspected before proceeding. The new audit matched
all 446 inherited hashes. Neither project root is a git repository; no clean git
state or commit is claimed. Existing experiments/checkpoints were not overwritten.
ResNet and all other training remain paused.

No complete real synchronized stereo dataset was found in project inputs or
inspected mounted directories. DexYCB is the preferred candidate (8 synchronized
calibrated RGB-D views, CC BY-NC 4.0, independent dataset hand annotations). Official
calibration (14,464 bytes) and one 640×480 RGB were retrieved. The annotation-only
mirror returns HTTP 401, `Invalid username or password.`, including on recheck.
No paired independent labels were obtained. The available intrinsics omit distortion
coefficients; explicit already-undistorted evidence was not established. Larger
archive range support was verified, but no full multi-GB dataset was downloaded.
InterHand2.6M and STB were investigated without establishing a qualified minimal
subset. This is a session-specific acquisition/qualification blocker, not a claim
that these datasets are universally inaccessible.

### 2. Coordinate and synchronization validation

DexYCB original extrinsics are camera-to-world, confirmed against the official loader;
the required world-to-camera interface must invert them exactly once and select the
calibration named by sequence metadata. No real GT projection, timing, scene coverage,
or eight-pair overlay check could be completed without paired labels.

New adapter checks include image hash/resolution versus K, proper rotation,
nonzero plausible metric baseline, GT projection, two-view triangulation cheirality,
explicit distortion processing, crop roundtrip, one left-hand reflection, sync
IDs/timestamps and sequence split isolation. Arbitrary nonrectified rigs do not
have a universal sign for raw horizontal disparity. Original sync provenance plus
visual review is required; matching IDs alone is insufficient.

### 3. Stereo optimization implementation

`scripts/run_real_e2.py` provides nominal-5000 pseudo extraction, real-K translation,
one shared left-camera MANO state with fixed camera transforms, and L2/Huber loss.
Both branches represent the same MANO inference under different camera interpretations.
Anisotropic K uses the geometric mean of crop fx/fy for scalar weak-camera depth;
this is a documented initialization heuristic, not a depth guarantee.

Loss averages weighted joints per view and then active views. Per-view confidence
is applied after joint normalization so it does not cancel. Pose/shape priors and
positive-depth penalties are included. GT enters only evaluation or explicit oracle
visibility. HaMeR has no confidence head; missing external predictions leave E2-5
unavailable. Missing oracle visibility leaves E2-6 unavailable; validity bits are
not substituted for visibility. Source/config/input/state/checkpoint/overlay/runtime
and compact trajectories have output paths; independent NumPy evaluation and scoped
SHA256 generation are wired into the runner.

### 4. Fixed experiment matrix

All rows are **NOT EXECUTED ON REAL INPUT**. Config: `configs/real_e2_v1.json`,
seed 42, 800 updates, same left-teacher initialization, decay at step 600.

| ID | Observations / variables | Real status |
| --- | --- | --- |
| E2-0 | Real-K initialization | Blocked |
| E2-1 | Left L2 / translation | Blocked |
| E2-2 | Left L2 / translation + orientation | Blocked |
| E2-3 | Stereo L2 / translation + orientation | Blocked |
| E2-4 | Stereo Huber / translation + orientation | Blocked |
| E2-5 | Stereo Huber + predicted confidence | Blocked; predictor also absent |
| E2-6 | Stereo Huber + oracle visibility | Blocked; visibility annotations absent |
| E2-7 | Stereo Huber / full pose | Blocked |
| E2-8 | Stereo Huber / full pose + shape | Blocked |
| E2-S0/S±1 | Correct versus offset synchronization | Not executed; neighboring-frame path not implemented |

### 5. Main results

There are **zero real E2 samples and no real accuracy/timing/failure-rate results**.
Do not fill the matrix with synthetic values. GPU 1 interface regression read the
existing 8 synthetic fixtures and saved optimized checkpoint, without any optimizer
updates or new synthetic scene. Run
`artifacts/real_e2_interface_regression_20261020_01` passed 13 checks; after the
per-view-confidence correction, `real_e2_interface_regression_20261020_02` reports
14 passed checks, checkpoint reconstruction maximum error **0 m**, and unchanged
source hashes. A repeated invocation of run02 was rejected with `FileExistsError`;
the existing completed run was inspected and retained, not overwritten or rerun.
Both regression runs explicitly set `real_e2_executed:false`.

Regression covers geometry/incorrect extrinsics/units/calibration/sync-annotation
rejection, crop/mirror roundtrips, L2/Huber reference agreement and gradients,
view-count normalization, missing-view masks, per-view weighting, existing checkpoint
reconstruction, and independent recomputation of archived absolute/relative MPJPE
and wrist depth. It does **not** establish real RGB inference or optimizer performance.

### 6. Failure samples and causes

No real sample has been optimized, so no real failure count or failure rate exists.
Acquisition failures are retained in `real_e2_audit_20261020_01/data_search/`.
The first downloaded RGB contains no usable visible hand for this purpose; it is
only a download/decoding probe, not a selected evaluation sample. Never exclude
future bad-pseudo samples based on optimizer outcomes. Nonfinite and quality failures
must stay in future per-sample reports with explicit denominators.

### 7. Meeting claims

| Claim/question | Evidence-supported conclusion |
| --- | --- |
| High-focal pseudo improves true stereo GT2D | Not tested on real stereo |
| Pseudo fitting improves absolute 3D and depth | Not established; earlier E1 showed metric-dependent outcomes |
| Stereo robust beats best monocular | Not tested on real stereo |
| Bad view can harm stereo L2 | Existing synthetic control demonstrates possibility only |
| Predicted confidence approaches oracle | Unknown; no qualified predictor/visibility inputs |
| Full pose/shape is best | Not supported by E1 mean absolute errors; real E2 unknown |
| ±1 frame offset is tolerable | Unknown |
| Meeting's real stereo improvement is supported | Not verified |

### 8. E3 gate

**Closed.** No automatic ResNet/KD training. Neither component regression nor
synthetic controls satisfies the real-E2 gate. Future acceptance requires a paired,
majority-sample improvement in independent absolute MPJPE or wrist position against
the best monocular configuration, without increased failure rates, under reasonable
robust loss and nonoracle confidence.

### 9. Remaining risks and next input

See `docs/real_e2_input_contract.md` for exact fields and runnable commands, and
`docs/real_e2_data_investigation.md` for retrieval provenance. Supply a legally usable
20–50-time stereo subset with corresponding RGB, timestamps/reliable sync mapping,
K, distortion or explicit undistortion provenance, metric extrinsics, physical
handedness, independent 21-joint GT and original IDs. Dataset-fitted MANO GT can still
have fitting bias and measurement noise; teacher pretraining overlap remains unknown.

The new RGB-to-teacher-to-optimizer path is executable but not end-to-end validated
on real inputs. Scene coverage, held-out sequence statistics, paired uncertainty,
sync-offset implementation, predicted confidence, and oracle visibility remain open.
No evidence from this delivery proves ResNet can replace ViT.

## Official DexYCB real E2 — completed matrix and synchronization sensitivity (2026-10-20)

This entry supersedes the historical acquisition blockers above. The fixed real
matrix actually ran; no synthetic control or earlier E1/RHD experiment was rerun.

### 1. Real E2 data and calibration

Downloaded the complete official subject-01 archive, **12,412,314,463 bytes**, and
official calibration, **14,464 bytes**, after checking approximately 263 GiB local
free space. Complete gzip CRC/EOF and all **5,314** selected subject/calibration
files passed original-archive comparison. Full subject remains local; selected
sequences are mirrored to `/workspace/pose_reconstract` on the GPU host.

- Subject official URL: https://drive.google.com/file/d/1Ehh92wDE3CWAiKG7E9E73HjN2Xk2XfEk
- Subject SHA256: `ce44e97a7567f7fb128b474ae33f327715314956a1c7b24cb08b08bc51cf4303`.
- Calibration official URL: https://drive.google.com/file/d/1UAwVKT4Rgb1fLcFoa1o71_-0NtSvvLAQ
- Calibration SHA256: `1f6cb30cfe290833ac66edec919c3b9b3f3937af2ba7bb78f0700a5349edd645`.
- Acquisition, HTTP confirmation responses and successful gdown calibration:
  `artifacts/dexycb_official_acquisition_20261020_01/`.

Official toolkit commit `64551b001d360ad83bc383157a559ec248fb9100` is installed and
used. Its subset-selection-only initializer diff is retained; its data interface
and field semantics are unchanged. [Data audit](dexycb_real_e2_data_audit.md)
provides exact official code references and selection details.

Selected **32 right-hand times**, 16 each from `20200709_150654` and
`20200709_150906`. Camera pairs are respectively
`836212060125 / 932122062010` (0.637785 m baseline) and
`932122060861 / 932122061900` (0.561535 m). All images are 640×480.
Official color/depth/label files, K, pose_m, joint_2d, joint_3d, MANO side and
serials are preserved in the manifest/raw labels. Detector boxes use pretrained
MediaPipe Hands 0.10.14 with 2× square extent, never GT boxes.

### 2. Coordinates and synchronization

Left camera is the shared reference. Official camera-to-master transforms give
`T_left_to_right = inverse(C_right) @ C_left`. Each camera retains its own GT;
right GT is additionally transformed into the left frame for shared-frame metrics.
No nonexistent extra world-GT field is required.

Maximum K-projection discrepancy is **0.001463 px**; cross-camera GT alignment
discrepancy is **2.95e-6 m**. The incorrect inverse produces at least **0.14707 m**
mean error per pair. K/resolution, positive depth, triangulation/disparity,
crop/full-image roundtrips and identity 21-joint mapping pass. Undistortion and
mirror counts are zero.

Distortion status is **dataset-delivered pinhole projection**, raw coefficients
unknown. Official K construction, pinhole renderer and camera-GT projection agree;
raw lens coefficients were not invented. These checks establish coordinate
consistency, not micrometer physical annotation accuracy.

Same official sequence/frame correspondence is used. Frame/30 is nominal time:
the recording guide uses an approximate synchronizer with `slop=0.1`, and actual
per-camera timestamps are unavailable. Eight GT pairs were inspected before
optimization; eight final paired overlays plus worst cases were inspected afterward.
See preparation `manifest_reviewed.json` and
`artifacts/dex_real_e2_visual_review_20261020_01/review.md`.

### 3. Stereo optimization and reproducibility

Each camera runs the same frozen HaMeR teacher. Nominal focal 5000 and real-K
translation are **two interpretations of the same MANO prediction**, not two
independent network predictions. Both views share one MANO pose, orientation,
shape and translation in the left-camera reference; camera K/R/t stay fixed.

Loss averages confidence-weighted joints within each view, then active views.
It supports L2/Huber, pose/shape priors and positive-depth penalties. E2-5 uses
non-GT epipolar-distance/image-bounds confidence, not learned visibility.
GT is used only for evaluation and audit in executed variants.

Seed 42; 800 Adam updates, 801 objective forward calls and zero closures for every
optimized state; fixed step-600 decay. Initialization is identical across variants.
All **256** persisted main states passed independent CPU MANO reconstruction,
camera immutability, frozen-variable and budget checks. Maximum CPU/GPU discrepancy:
**1.1921e-7 m**; in-run checkpoint reload discrepancy is zero.

Main source/config/state/overlay/trajectory snapshots and scoped hashes are in
`artifacts/dex_real_e2_matrix_20261020_02/`. Independent NumPy metrics are in
`independent_metrics.json`; CPU state verification and repository commits/diffs,
dependency versions and source hashes are in
`artifacts/dex_real_e2_postprocess_20261020_01/`. The project root is not a git
repository; no clean root commit is claimed. Physical GPU 1 was used.

### 4. Complete experiment matrix

| Variant | Observations / variables | Status |
| --- | --- | --- |
| E2-0 | Shared real-K initialization | 32 evaluated, zero updates |
| E2-1 | Left L2 / translation | 32 completed |
| E2-2 | Left L2 / translation + orientation | 32 completed |
| E2-3 | Stereo L2 / translation + orientation | 32 completed |
| E2-4 | Stereo Huber / translation + orientation | 32 completed |
| E2-5 | Stereo Huber + non-GT geometry confidence / translation + orientation | 32 completed |
| E2-6 | Oracle joint visibility | Unavailable; official visibility not established |
| E2-7 | Stereo Huber / full pose | 32 completed |
| E2-8 | Stereo Huber / full pose + shape | 32 completed |

E2-S0/S±1 uses 30 common times with both neighboring right detections. The two
sequence-final times lack +1 frames and are excluded from all three sensitivity
conditions. S0 reuses main E2-4; original GT and initialization are unchanged.
Completed `artifacts/dex_real_e2_sync_20261020_01/`: 60 new optimized states, with
independent evaluation and checkpoint reconstruction passed (maximum discrepancy
1.1921e-7 m). Additional correspondence verification confirms exact original GT,
initialization, K/T and left pseudo preservation, plus correct right camera,
sequence, ±1 index, image SHA256 and saved neighbor-teacher targets.

| Common 30-time subset | S0 (cached E2-4) | S−1 | S+1 |
| --- | --- | --- | --- |
| Abs MPJPE mean / median / P95 mm | 10.007 / 8.286 / 16.941 | 17.727 / 11.128 / 51.014 | 14.749 / 10.671 / 31.664 |
| Left depth mean mm | 11.803 | 18.906 | 12.510 |
| Right depth mean mm | 12.682 | 19.302 | 12.358 |
| Pseudo residual mean px | 2.562 | 3.200 | 3.922 |
| Quality failures | 1/30 | 3/30 | 1/30 |
| Absolute error worse than S0 | — | 21/30 | 27/30 |

Both offsets have zero numerical/positive-depth failures. Mean paired absolute
degradation is **+7.720 / +4.742 mm**; median degradation **+2.523 / +3.205 mm**.
All metrics and per-sequence/per-sample deltas are in
[`sync results`](../artifacts/dex_real_e2_sync_report_20261020_01/results.md).
One frame is nominally 33.333 ms; these are frame-index interventions, not measured
physical camera-latency experiments. The main run's sealed status describes its
own scope only; subsequent sensitivity results live in the separate run.

### 5. Main results

Mean values below; full **mean / median / P95**, each camera's absolute MPJPE,
shared-frame MPJPE, 2D EPE, wrist position, depth, variable changes, optimizer
counts, timing and allocation are in
[`results.md`](../artifacts/dex_real_e2_report_20261020_01/results.md).
Every sample is retained in the independent JSON.

| Variant | Abs MPJPE mm | Root-relative mm | Left depth mm | Right depth mm | Pseudo residual px | Quality failures | Backend mean s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E2-0 | 41.911 | 6.762 | 39.355 | 29.896 | 14.236 | 18/32 | 0.005 |
| E2-1 | 35.930 | 6.762 | 35.164 | 31.841 | 9.486 | 10/32 | 3.987 |
| E2-2 | 24.712 | 18.249 | 17.273 | 17.105 | 5.511 | 4/32 | 9.421 |
| E2-3 | 9.931 | 14.745 | 11.731 | 12.734 | 2.564 | 1/32 | 9.413 |
| E2-4 | 9.836 | 14.207 | 11.399 | 12.223 | 2.523 | 1/32 | 9.559 |
| E2-5 | 9.818 | 13.711 | 11.010 | 12.008 | 2.533 | 1/32 | 9.629 |
| E2-7 | 9.682 | 12.318 | 9.069 | 10.118 | 2.074 | 1/32 | 10.528 |
| E2-8 | 9.762 | 12.265 | 8.501 | 9.511 | 1.886 | 1/32 | 11.321 |

All variants have **0/32 numerical and positive-depth failures**. Quality thresholds
were fixed at crop GT2D >20 px, relative MPJPE >50 mm, or left wrist depth >100 mm.
Peak allocated memory is approximately 2.635 GiB including the resident teacher.
Backend time excludes teacher inference, serialization and reload; it is not FPS.

Best predefined left-only baseline is E2-2, selected globally by mean absolute
MPJPE, not separately per sample. E2-4 improves **28/32 (87.5%)** absolute errors,
**14/16 in each sequence**. Mean paired improvements are **16.090 / 13.661 mm**
in the two sequences. E2-5 also improves 28/32, with sequence improvements
16.306 / 13.481 mm. Quality failure rate falls from 12.5% to 3.125%.
These descriptive results pass the requested majority/failure gate on this subset.
Median paired E2-4 improvements are 12.615 / 2.104 mm in the two sequences:
the magnitude is heterogeneous even though each sequence improves in 14/16 times.

Depth is less uniform: E2-4 improves left depth in only **17/32** samples.
Sequence 150654 worsens by **1.445 mm** on average (6/16 improve), while 150906
improves by 13.193 mm (11/16 improve). Root-relative MPJPE improves versus E2-2,
but **worsens versus the original MANO prediction, 6.762 → 14.207 mm**.

The separate zero-inference camera-branch audit compares each camera's **own same
MANO** under both camera interpretations. Nominal-target GT2D improves from
**10.026 → 5.279 px left** and **5.205 → 4.199 px right**, each in **26/32**
samples. See `artifacts/dex_real_e2_branch_audit_20261020_01/`. The main table's
right shared-initialization EPE (20.532 px) also changes the source MANO/view and
must not be used as a pure right-camera focal comparison.

### 6. Failures and causes

The single E2-4 quality failure is `20200709_150906_000004`: blurred image-edge
hand, absolute MPJPE 32.494 mm and crop GT2D 21.639 px, despite pseudo residual
2.789 px. Blur/edge conditions are observed, not causally isolated.

Worst stereo regression is `20200709_150906_000017`. E2-4 reduces pseudo residual
**2.256 → 1.198 px**, but increases absolute error **7.986 → 16.105 mm** versus
E2-2. Right GT2D worsens 2.112 → 3.534 px. E2-5 remains at 16.230 mm.
Overlays show biased pseudo joints; geometrically plausible targets can still
be inaccurate. No failure or degraded sample was removed.

Preparation run01's invalid-annotation failure and main run01's MANO batch-shape
failure are retained; successful preparation/main runs use distinct run02 IDs.
Final state verification performs zero optimizer updates.

### 7. Meeting claims and required corrections

| Claim | Evidence-supported conclusion |
| --- | --- |
| High-focal pseudo improves real-K GT2D | Supported on average and 26/32 samples per camera; not absolute accuracy or every sample |
| Real-K initialization guarantees correct depth | Rejected; initial left mean depth error is 39.355 mm |
| Pseudo fitting also improves true absolute 3D | Supported on this subset for stereo; individual counterexamples remain |
| Stereo robust beats best predefined left-only baseline | Supported descriptively: 24.712 → 9.836 mm, 28/32 improve, both sequences, fewer quality failures |
| L2 is badly harmed by bad views here | Not established: L2 is 9.931 mm versus Huber 9.836 mm, with the same failure count |
| Nonoracle confidence preserves stereo benefit | Supported; E2-5 is 9.818 mm, but only 0.018 mm better than uniform Huber |
| Predicted confidence approaches oracle visibility | Unknown: the heuristic is not learned visibility and oracle E2-6 is unavailable |
| Translation + orientation always beats full pose/shape | Not supported here: full pose is 9.682 mm and improves wrist metrics; adding shape is not best for absolute MPJPE |
| Lower pseudo residual means universally better pose | Rejected by the regression example and root-relative deterioration versus raw |
| ±1-frame synchronization is harmless | Rejected on this subset: common-time absolute MPJPE 10.007 → 17.727 / 14.749 mm, with 70% / 90% worsening |
| Real stereo joint reprojection works well | Supported for absolute position on these detected, calibrated samples; not universal depth, articulation or deployment performance |

### 8. E3 gate

The descriptive real-E2 gate **passes on this subset**, including nonoracle E2-5.
This permits proposing a limited E3 comparison, not declaring a small model
equivalent to ViT. **No E3/ResNet/KD training has started.**
Before a future E3, freeze one backend and compare ViT/ResNet raw and optimized
on identical data, reporting both absolute and root-relative metrics and
`delta_after` versus `delta_before`. Do not assume lower training loss proves
backend compensation.

### 9. Remaining risks and evidence limits

One subject, two temporally correlated sequences and one bowl interaction are
insufficient for population significance or broad scene/occlusion generalization.
There are no sequence-level uncertainty estimates. Detector misses and missing
annotations are outside the backend denominators; selection was before inference.
GT has dataset MANO-fitting bias; teacher pretraining overlap is unknown.
Physical subframe timestamps and oracle visibility are unavailable.

“Best monocular” refers to the two preregistered **left-only optimized variants**;
right-only optimization was not tested. Right teacher's own raw real-K absolute
error is 21.475 mm, recorded in the branch audit. The pure focal audit and shared
left-reference stereo matrix must remain distinct.

No deployment speed claim, universal pose/depth improvement, or ResNet replacement
claim follows from this run. Root-relative deterioration deserves attention before
using the backend for articulation-sensitive downstream tasks.

Final scoped SHA256 manifest:
`artifacts/dex_real_e2_delivery_sha256_20261020_01.json`.
Local-versus-remote and inherited-evidence verification:
`artifacts/dex_real_e2_delivery_verification_20261020_01.json`.
The 12.4 GB archive remains local and is hashed separately in acquisition verification;
the delivery manifest covers shared experiment artifacts, selected inputs and code/docs.

## Real E2 pose-preserving backend ablation

**Executive summary (2026-10-20):**
- Translation-only B2 exactly preserves HaMeR global orientation, articulated pose and shape.
- B2 reaches **8.125 mm absolute MPJPE** while retaining the original **6.762 mm wrist-relative MPJPE**.
- B2 has 0/32 numerical/depth failures, 1/32 quality failures, and improves absolute error in 27/32 times versus B0.
- Rotation-aligned diagnostic stays 4.038 mm for B0/B2/B4: B4's 14.207 mm wrist-relative error comes from global rotation, not changed finger articulation.
- The fixed B5-B8 anchors are too weak at this loss scale; λ=1 still leaves 7.996° mean rotation change and 13.799 mm wrist-relative error.
- No orientation anchor is needed because B2 already satisfies all fixed-backend thresholds.
- Recommended future-E3 backend: **B2, stereo Huber + translation only**, fixed globally.
- The backend-selection gate is open for a future E3, but **E3/ResNet/KD remain paused and were not started**.

### Protocol and diagnostics

Run: `artifacts/dex_pose_preserving_20261020_01`. It reads only the sealed
`dex_real_e2_matrix_20261020_02` states: zero HaMeR calls, no sample reselection,
and no GT-conditioned optimization. Ordered sample IDs, real-K initialization,
pseudo targets, K/R/t, GT, seed 42 and 800-update budget are unchanged. B0/B4/B10
copy E2-0/E2-4/E2-7 with zero new updates. E1, RHD, synthetic E2 and the full real
E2 matrix were not rerun.

Config: `configs/pose_preserving_v1.json`. The existing pose prior remains
**0.1 × mean rotation-matrix element squared difference**. B5-B8 separately add
`λ_rot × geodesic(R_global,R_initial)^2` in radians squared at fixed
λ={0.001,0.01,0.1,1}. B9 freezes global orientation/shape and opens articulated
pose with the existing 0.1 prior. No post-hoc anchor was added because B2 passes.

Wrist-relative MPJPE removes wrist translation only. Rotation-aligned diagnostic
applies one SO(3) Kabsch rotation to 20 non-wrist joints, with no scale, reflection
or translation; it never replaces absolute accuracy. Bone angles use wrist→MCP
and consecutive finger bones. Fingertips are joints 4/8/12/16/20.

DexYCB `pose_m` was decoded as global axis-angle + 45 MANO PCA coefficients +
translation using official subject betas. Reconstruction over 672 joints is
0.722 mm mean / 0.000063 mm median / 3.752 mm P95. Wrist/MCP anchors average
0.000050 mm with maximum below 0.0001 mm. The 5.05 mm maximum at vertex-derived
tips is an implementation tip convention. This validates global orientation
coordinates; its error is available rather than guessed.

### Main results

Cells are mean / median / P95. Full 2D, depth, fingertip, bone, runtime, variable
change and per-sample tables are in
`artifacts/dex_pose_preserving_report_20261020_final_v2/results.md`.

| Variant | Variables | Absolute mm | Wrist-relative mm | Rot-aligned diagnostic mm | Global orientation deg | Quality |
| --- | --- | --- | --- | --- | --- | --- |
| B0 | none; E2-0 | 41.911 / 37.363 / 78.888 | 6.762 / 5.382 / 9.997 | 4.038 / 4.033 / 5.755 | 5.871 / 5.174 / 10.358 | 18/32 |
| B1 | left Huber, translation | 38.056 / 29.054 / 90.046 | 6.762 / 5.382 / 9.997 | 4.038 / 4.033 / 5.755 | 5.871 / 5.174 / 10.358 | 10/32 |
| **B2** | **stereo Huber, translation** | **8.125 / 6.254 / 15.307** | **6.762 / 5.382 / 9.997** | **4.038 / 4.033 / 5.755** | **5.871 / 5.174 / 10.358** | **1/32** |
| B3 | stereo L2, translation | 8.132 / 6.224 / 14.933 | 6.762 / 5.382 / 9.997 | 4.038 / 4.033 / 5.755 | 5.871 / 5.174 / 10.358 | 1/32 |
| B4 | stereo Huber, trans+orient; E2-4 | 9.836 / 8.286 / 16.789 | 14.207 / 12.518 / 29.479 | 4.038 / 4.033 / 5.755 | 10.760 / 9.748 / 21.266 | 1/32 |
| B5 | B4 + λ=.001 | 9.836 / 8.286 / 16.789 | 14.207 / 12.517 / 29.477 | 4.038 / 4.033 / 5.755 | 10.760 / 9.748 / 21.264 | 1/32 |
| B6 | B4 + λ=.01 | 9.835 / 8.285 / 16.787 | 14.203 / 12.514 / 29.460 | 4.038 / 4.033 / 5.755 | 10.757 / 9.747 / 21.249 | 1/32 |
| B7 | B4 + λ=.1 | 9.823 / 8.273 / 16.772 | 14.167 / 12.485 / 29.295 | 4.038 / 4.033 / 5.755 | 10.732 / 9.734 / 21.098 | 1/32 |
| B8 | B4 + λ=1 | 9.700 / 8.164 / 16.630 | 13.799 / 12.194 / 27.444 | 4.038 / 4.033 / 5.755 | 10.473 / 9.612 / 19.451 | 1/32 |
| B9 | translation+pose, global frozen | 9.095 / 7.857 / 15.629 | 8.935 / 7.058 / 12.199 | 5.258 / 4.914 / 8.560 | 5.871 / 5.174 / 10.358 | 1/32 |
| B10 | trans+orient+pose; E2-7 | 9.682 / 8.432 / 15.892 | 12.318 / 10.630 / 24.892 | 5.170 / 4.692 / 8.900 | 9.910 / 8.782 / 19.247 | 1/32 |

All variants have 0/32 numerical and positive-depth failures. B2 left/right wrist
depth is 6.476/6.685 mm; wrist position 8.847 mm. Five-tip absolute/wrist-relative
error is 9.660/9.326 mm. Left/right GT2D EPE is 3.656/3.471 px; pseudo residual
is 3.555 px. B2 changes translation by 42.358 mm and changes every rotation and
shape value by exactly zero. Runtime is 4.218 s P50 / 4.353 s P95 versus B4's
9.376/10.134 s; these are offline backend times, not FPS.

B2 preserves B0 bone errors exactly: thumb 5.688°, index 6.061°, middle 6.258°,
ring 6.988°, little 8.777°. B4 increases these to
8.267°/8.861°/8.026°/9.584°/11.403°. B9 changes articulated rotations by
22.824° and worsens rotation-aligned diagnostic to 5.258 mm. Finger-pose
optimization is neither necessary nor internally benign here.

### Paired results and selection

Exact per-sample candidate/baseline/deltas are in
`independent_metrics_final.json`; improvement counts ignore <1e-4 numerical noise.

- B2 vs B0: absolute −33.785 mm, 27/32 improve; wrist-relative,
  rotation-aligned, fingertip-relative and orientation are unchanged.
- B2 vs E2-2: absolute −16.587 mm (31/32), wrist-relative −11.487 mm
  (27/32), global orientation error −6.961° (30/32).
- B2 vs B4: absolute −1.711 mm (27/32), wrist-relative −7.445 mm (29/32),
  orientation error −4.890° (29/32).
- B8 vs B2: absolute +1.575 mm, wrist-relative +7.037 mm; only 5/32 and
  3/32 improve. B8 is the best fixed anchor but is dominated by B2.
- B10/E2-7 vs B2: absolute +1.556 mm, wrist-relative +5.556 mm, and
  rotation-aligned diagnostic +1.131 mm. Full pose is not needed.

At λ=1, mean weighted anchor is 0.027 versus reprojection 0.922; B8 still changes
orientation by 7.996°. The fixed anchor sweep is weak at this loss scale. No
post-hoc anchor is required. B2 and B3 form the numerical absolute/wrist-relative
frontier; the preregistered preference for robust loss selects **B2**.

### Meeting interpretation and E3 decision

The stereo gain primarily corrects **root translation and absolute depth**.
E2-4 never changed articulated pose or shape: its degradation comes from an
8.370° global rotation change, confirmed because Kabsch returns the same 4.038 mm
diagnostic as B0/B2. “2D accurate + backend solves 3D” is supported here for
calibrated multi-view root translation, wrist position and depth. It is not
evidence that the backend reliably fixes global orientation or finger articulation;
opening those variables worsens their independent errors.

Future-E3 fixed backend: **B2 Stereo Huber + translation only**, applied uniformly
without GT switching. It meets absolute ≤12 mm, wrist-relative ≤8 mm and failure
≤1/32. The backend-selection gate is open, but this is the required submission
point: **ResNet/KD/E3 were not started**.

### Verification and preservation

`artifacts/dex_pose_preserving_verification_20261020_final.json` independently
verifies all **352 states** (256 new, 96 reused), ordered IDs, source hashes,
cached initialization/pseudo/K/T identity, frozen variables, checkpoint reload,
800-update budgets, positive finite results, and metric equality. Maximum checkpoint
reconstruction discrepancy is 1.1921e-7 m.

The initial evaluator's raw `acos` showed ~0.008° noise for exactly frozen matrices.
It is preserved as `independent_metrics.json`; the final evaluator sets bitwise-equal
rotations to zero. All intermediate metrics and failed verification logs remain,
but `independent_metrics_final.json` is authoritative. Final manifest:
`artifacts/dex_pose_preserving_sha256_20261020_final_v3.json`; local/remote verification:
`artifacts/dex_pose_preserving_local_verification_20261020_final_v3.json`.

## E3 Fast-HaMeR checkpoint audit and retargeting baseline (2026-10-20)

Formal E3 is **blocked by the missing author-trained ResNet checkpoint**.
Current official Fast-HaMeR main matches downloaded commit
`009821c80e9bfa4e111c6b9b641408126c53d03f`; releases/tags are empty.
README asks for user-supplied student weights; downloads point to the original
HaMeR ViT teacher. ResNet-50/ImageNet initialization and supervised/KD configs
are present, but are not trained hand-reconstruction checkpoint evidence.
Random or 128-image RHD pilot weights were not used.

Frozen B2 and all old experiments remain unchanged. Cached raw B0 has absolute
MPJPE **41.911 mm**; B2 has **8.125 mm**, wrist-relative **6.762 mm**,
and **1/32** quality failures. These authoritative cached B0 metrics replace
neither old files nor other camera-convention results; the earlier conversational
24.71 mm raw summary is not used in this E3 reference table.

No E3-C/D, compensation ratio, speedup or memory claim is available. Historical
32-sample runtime is not the requested E3 20-warm-up/100-iteration benchmark.
No teacher rerun, backend rerun, training, YOLO, TensorRT or VLA was started.

All 32 ViT+B2 outputs are retained for retargeting in
`artifacts/e3_vit_b2_retargeting_20261020_03/retargeting.json` and
`retargeting.npz`, with camera-frame and wrist-local joints in meters, frozen
MANO rotations/shape, identities, nominal timestamps, K/T metadata and flags.
Confidence is unavailable; GT-assisted quality flags are evaluation-only,
and the failed-quality sample remains included.

Export verification passed: exact cached predictions and frozen variables,
ordered IDs, unchanged calibration, finite positive depth, JSON/NPZ round-trip
and source hashes. Inference and optimization calls: **0 / 0**.
Full audit, metric matrix, checkpoint gap and training-data requirements:
`docs/e3_fasthamer_audit_report.md`; schema: `docs/e3_retargeting_schema.md`.
This completes the requested stop point; ViT+B2 remains the baseline.

## ResNet-50 no-KD training and fixed E3 (2026-10-20)

The earlier checkpoint blocker was resolved by training the requested model from
scratch: an ImageNet1K V2 initialized torchvision ResNet-50 backbone with the
Fast-HaMeR MANO transformer head initialized from scratch. No teacher, KD,
adversarial loss, or pilot checkpoint was used. The complete usable RHD set was
split into 61,895 train instances, 2,751 development instances, and 1,183 test
instances using the official train/evaluation boundary and disjoint evaluation
image IDs. The fixed 32-pair DexYCB E3 set was excluded from training and model
selection.

R0 passed archive CRC, annotation, handedness, unit, joint-order, crop-affine,
and 3D-to-2D projection checks; maximum projection discrepancy was 0.239 px.
R1 completed 500 real-loader updates after preserving and fixing an augmentation
failure found at step 156. Strict checkpoint reload was exact, both backbone and
head updated, and finite/depth/projection-collapse gates passed.

R2 stopped by the preregistered six-evaluation patience rule after 46,416 updates
(25 epochs). The frozen best-development checkpoint is step 34,812, selected
without E3 access:

- wrist-relative MPJPE: **32.119 mm**
- rotation-aligned diagnostic MPJPE: **19.125 mm**
- visible 2D EPE: **15.712 px**
- fingertip wrist-relative error: **43.076 mm**
- nonfinite / negative-depth / projection-collapse / total failure: **0**
- checkpoint SHA256:
  `95dd65e7724f1fbef33ec8b2fcf9f491c948d990061fc67d0f533e73ce32ea0a`

Strict reload discrepancies are exactly zero. Batch-1 network+MANO latency on
the training benchmark was 8.660 ms mean (20 warm-up, 100 synchronized runs).

### One-shot fixed E3

The frozen checkpoint was evaluated once on the original ordered 32 DexYCB
stereo pairs. Images, crop affine, K/R/t, GT, synchronization, nominal focal
5000, and B2 hyperparameters were unchanged. B2 optimized translation only for
800 steps; global orientation, articulated pose, and shape remained
exact-equality frozen.

| Metric | ViT raw | ViT+B2 | ResNet raw | ResNet+B2 |
| --- | ---: | ---: | ---: | ---: |
| Absolute MPJPE (mm) | 41.911 | 8.125 | 124.331 | 31.275 |
| Wrist-relative MPJPE (mm) | 6.762 | 6.762 | 33.185 | 33.185 |
| Rotation-aligned diagnostic (mm) | 4.038 | 4.038 | 23.890 | 23.890 |
| Fingertip wrist-relative error (mm) | 9.326 | 9.326 | 48.711 | 48.711 |
| Left/right frontend GT-2D EPE (px) | 5.279 / 4.199 | same | 12.138 / 15.771 | same |
| Quality failures | 18/32 | 1/32 | 27/32 | 21/32 |

Raw degradation is 82.420 mm, optimized degradation is 23.150 mm, and ResNet
backend recovery is 93.056 mm. The compensation ratio is **71.9%**. This is
strong evidence that frozen B2 recovers much of the small model's global
translation/depth error, but it does not support replacement: ResNet+B2 exceeds
ViT+B2 by 23.150 mm absolute, 26.423 mm wrist-relative, and 39.385 mm fingertip
wrist-relative, with 21/32 quality failures. Projection-collapse,
nonfinite, and depth-failure gates all pass, so the failure is model geometry and
cross-dataset generalization rather than numerical collapse.

The meeting hypothesis is therefore **partially supported for absolute
translation recovery and not supported for replacing ViT with this RHD-only
no-KD model**. Translation-only B2 cannot repair global orientation or finger
articulation. The miss is far beyond the "small miss" condition for a controlled
KD follow-up, so KD was not started. A future frontend iteration requires broader
in-domain training and stronger pose supervision before repeating a separately
preregistered E3; this frozen E3 must not be reused for checkpoint selection.

Independent verification passed for all 32 samples: ordered IDs and source rows,
GT/K/T/affine identity, exact frozen rotations/betas, finite positive depth,
sample and aggregate metric recomputation, checkpoint hash, and artifact
manifest. Verification made zero inference and optimization calls. Authoritative
artifacts:
`artifacts/resnet50_rhd_r2_formal_20261020_01/results.json` and
`artifacts/resnet50_fixed_e3_20261020_01/{results.json,results.md,verification.json}`.

## Real-mixed no-KD M-S0 subset smoke (2026-10-21)

M-S0 used the preregistered available-domain mixture without teacher, KD,
adversarial loss, or RHD checkpoint initialization. Every 30-example batch
contained 16 DexYCB subject-02 views, 10 FreiHAND captures, and 4 RHD examples.
The run is a loader/loss/checkpoint smoke and is not an accuracy result or a
checkpoint-selection evaluation.

The official DexYCB subject-02 archive passed complete gzip CRC consumption,
SHA-256 verification (`df785049fdaf9266cbba4102084bd435e72ae12836fa603bdce00182c449bd89`),
and metadata audit for all 100 sequences. Metadata-only uniform selection
produced 2,000 synchronized times and 4,000 views. The extracted manifest
SHA-256 is
`d9e64467abaab92310d0fac5c5ff60b6373b18e913b1eefe583ab36a6b8c81a4`.
A separate 64-view coordinate audit passed with 1.38e-5 px mean, 2.78e-5 px
P95, and 4.69e-5 px maximum 3D-to-2D projection error. Crop/full-image
round-trip error was at most 1.61e-13 px.

The training loader used independent validity masks. DexYCB provided metric
2D/3D and valid global orientation; its PCA articulated pose and subject betas
were masked pending separate convention audits. FreiHAND provided 2D/3D,
global orientation, articulated axis-angle pose, and betas. Its final three fit
values were correctly treated as `uv_root + scale`, not xyz translation. RHD
provided visible 2D and metric wrist-relative 3D only. The single 2D term is
the Fast-HaMeR weak-camera reprojection output; it was not duplicated as a
second camera loss.

Run `resnet50_real_mixed_m_s0_subject02_20261021_01` completed 500/500 updates:

| Domain | Wrist-relative MPJPE (mm) | Fingertip error (mm) | Valid 2D EPE (px) | Projection collapse |
| --- | ---: | ---: | ---: | ---: |
| DexYCB subject-02 subset | 17.929 | 26.350 | 28.975 | 0 |
| FreiHAND | 48.019 | 69.287 | 31.474 | 0 |
| RHD | 50.620 | 64.731 | 32.079 | 0 |

All seven loss components had nonzero effective sample/scalar counts and
nonzero gradients to the backbone plus the relevant decoder. Final nonfinite
and negative-depth rates were zero in every domain. Strict checkpoint reload
differences were exactly zero; backbone and pose-head parameters both changed.
The last checkpoint SHA-256 is
`f3048016e1c9780ec248d7a5fd88fa368fcf4aa6b1b75df4eb204f1df5e24e98`.

FreiHAND showed a transient 9.375% projection-collapse rate at the step-100
snapshot, recovering to zero at steps 250 and 500. M-S0 therefore passes the
final plumbing gate, but this transient instability must inform the M-S1
optimizer schedule; M-S1 must not treat the 500-step subset metrics as a model
selection result.

Authoritative evidence:
`artifacts/resnet50_real_mixed_m_s0_subject02_20261021_01/{results.json,provenance.json,evaluations.jsonl,steps.jsonl}`
and `artifacts/dexycb_subject02_coordinate_audit_20261021_01/audit.json`.

## Real-mixed no-KD M-S1 preparation (2026-10-21)

The M-S1 protocol uses all annotation-valid available DexYCB subject-02 views
for training, subject-07 for independent development, and the previously sealed
subject-08/09 manifests for Final-E4. No Final-E4 inference is authorized in
this stage. FreiHAND groups all four RGB variants of each capture into the
same split; its released annotations do not support a subject/sequence
independence claim. RHD retains its existing image-disjoint split.

Complete subject-02 extraction contains 115,872 RGB/label files. Fixed
annotation checks retained 7,134 synchronized times and 55,762 views, rejecting
2,174 views with reasons recorded in the manifest. Its sealed manifest SHA-256
is `d67e712b44b191c4b7f6b414c7d849d9ec5624b72484a33665cf771339f8203c`.
The first preparation attempt failed on YAML tuple loading; the error log is
retained and validation was rerun after correcting the loader.

Preregistered configuration:
`configs/resnet50_real_mixed_m_s1_subject02.json`. The run starts from
ImageNet1K V2 and a new MANO head, uses AdamW at 3e-5 with 500 warm-up updates,
cosine decay, gradient clipping at 1.0, and at most 10,000 updates. Full
independent dev evaluation occurs every 1,000 updates. Two consecutive
evaluations with projection collapse stop training; dev patience is five
evaluations. The same full dev will also evaluate the frozen RHD-only baseline.
Checkpoint selection uses independent DexYCB dev only, with zero failure
required in every domain. M-S0's step-100 FreiHAND 9.375% collapse remains part
of the failure history.

This entry records preparation, not completed training or an accuracy result.
The final M-S1 report must establish whether independent real dev improves
before any M-S2 eligibility conclusion.

Subject-07 preparation retained 5,957 synchronized times and 47,590 views
from all 100 sequences. The 11,002 excluded views comprise 10,936 with
nonfinite/nonpositive 3D labels and 66 with fewer than four valid 2D joints.
There were no projection-error exclusions. Mean source reprojection
disagreement is 1.48e-5 px. The development manifest SHA-256 is
`8f8f9f57a80b9dd7f547f8a9ec9b6788fe4a4f9df2a48f366c7e19e27a6af3ea`.
These annotation-based exclusions precede any model access.

All remote inputs were synchronized and the three split manifests were frozen
before inference. The frozen training inventory has 234,733 samples:
55,762 DexYCB, 117,076 FreiHAND, and 61,895 RHD. Full independent development
has 63,505 samples: 47,590 DexYCB, 13,164 FreiHAND, and 2,751 RHD.
Final-E4 contains 1,000 previously sealed views; only its manifest was read.
The split-file SHA-256 values are:

- Train: `c0aa72466cf8e1326bb1a4bb6dc5e27b3a04abc1eb4c829f79238b2842c176ad`
- Dev: `971d0a7d45e736955ba0836672ee7979dac87eb4829d6e24248bebf5779be772`
- Final-E4: `41f72e3ef1733998afa701faa9aec68ca2e68244ac4268a8526007708fc9b877`

The pretraining coordinate audit passed all 384 samples (64 per domain and
train/dev split). Visual spot checks of the saved DexYCB, FreiHAND and RHD
overlays found no obvious coordinate or handedness mismatch. Audit evidence:
`artifacts/m_s1_coordinate_audit_20261021_01/`. Split evidence:
`artifacts/m_s1_frozen_splits_20261021_01/`. These are preparation checks,
not model accuracy evidence.

The full-dev RHD-only baseline reproduces the historical 32.119 mm wrist,
43.076 mm fingertip, and 15.712 px 2D values on **RHD dev**. On the frozen
DexYCB subject-07 dev, the same checkpoint instead yields 64.326 mm,
84.551 mm, and 29.197 px. The historical values remain the requested absolute
gates; paired improvement claims use the same independent DexYCB dev.

After four completed M-S1 evaluations, DexYCB dev improved monotonically,
but all three RHD metrics still exceeded their same-dev baseline by more than
25%. A single conditional sampler correction was registered in
`configs/resnet50_real_mixed_m_s1_sampler_correction.json`: batch counts change
from 16/10/4 to 12/10/8 for DexYCB/FreiHAND/RHD. Total batch size, seed, fresh
initialization, labels, losses, optimizer, schedule, budget, selection and
stopping rules remain identical. The correction runs only if the completed,
verified source run still has severe domain imbalance at its selected
checkpoint and improves DexYCB wrist error over the same-dev baseline,
without a collapse/nonfinite stop. It starts fresh, retains all source-run
evidence, and must reproduce all three frozen split-file hashes before any
model access. This is an M-S1 correction, not M-S2 or KD.

### Completed first M-S1 run

`resnet50_real_mixed_m_s1_subject02_20261021_01` completed 10,000 updates.
The frozen best-dev checkpoint is update 9,000; the independent saved-artifact
verifier passed and both checkpoint reload checks had exactly zero forward
difference.

| Independent dev domain | Model | Wrist MPJPE mm | Fingertip mm | Visible 2D px |
|---|---|---:|---:|---:|
| DexYCB subject07 | RHD-only | 64.326 | 84.551 | 29.197 |
| DexYCB subject07 | M-S1 best | 31.477 | 43.151 | 20.170 |
| FreiHAND | RHD-only | 55.339 | 76.477 | 19.475 |
| FreiHAND | M-S1 best | 26.829 | 42.664 | 14.597 |
| RHD | RHD-only | 32.119 | 43.076 | 15.713 |
| RHD | M-S1 best | 42.020 | 54.484 | 24.205 |

DexYCB paired sequence-bootstrap 95% intervals for the change are
[-34.734, -31.003] mm wrist MPJPE and [-43.754, -39.155] mm fingertip error,
using 100 sequence clusters within the one independent development subject.
This supports substantial improvement on this real dev set. It does not
establish performance on new subjects beyond this dev subject.

Every full M-S1 dev evaluation had zero collapse, negative-depth and nonfinite
rates. Eighteen training updates between 281 and 581 had batch collapse,
with a maximum batch rate of 6.667%; those records and the earlier M-S0
step-100 failure remain preserved.

M-S2 eligibility is **false**: wrist/fingertip do not pass the preregistered
10% margins below the historical absolute thresholds, and RHD has severe
regression. RHD received only 40,000 draws from 61,895 eligible samples.
The preregistered sampler-correction condition is therefore satisfied.

Checkpoint SHA-256:
- Best-dev: `1d0457e066cbd2057d7492eafb25a17dc4b0d086f4a0fdb79346baedc3331a89`
- Last: `e1ae9fd6458290b26b95a9a56e5e40eae15e84e0b39d7ec94eee437692c8ad03`

Full evidence is in
`artifacts/resnet50_real_mixed_m_s1_subject02_20261021_01/`.
No E3, Final-E4 inference, KD, M-S2, B2 tuning, integration or retargeting was run.

### M-S1 registered sampler correction

The verified first-run result triggered the one permitted correction:
`resnet50_real_mixed_m_s1_sampler_correction_20261021_01`, launched on
2026-10-22. The runner verified identical train/dev/Final-E4 split-file hashes
and repeated the coordinate audit successfully (384 samples) before model
access. Batch composition is now DexYCB/FreiHAND/RHD 12/10/8. This is fresh
training, with no first-run checkpoint initialization.

The detached pipeline in `scripts/run_m_s1_sampler_correction.sh` performs
training, report generation and saved-artifact verification. It has no action
to launch another experiment. The original run and its failure records are
retained. Final correction metrics will be appended after completion and
verification; this entry does not claim completion.

### Completed M-S1 sampler correction and final comparison

`resnet50_real_mixed_m_s1_sampler_correction_20261021_01` stopped with `budget_completed` at 10000 updates; frozen best-dev is update 10000. Both runs passed local saved-artifact verification, including checkpoint hashes and ordered full-dev sample IDs.

| Independent dev | Wrist MPJPE mm | Fingertip mm | Visible 2D px | Failure |
|---|---:|---:|---:|---:|
| dexycb | 30.661 | 42.100 | 19.653 | 0.0000% |
| freihand | 26.186 | 41.993 | 14.184 | 0.0000% |
| rhd | 38.500 | 50.203 | 22.754 | 0.0000% |

Same-dev DexYCB improvement over the frozen RHD-only checkpoint: wrist 52.34%, fingertip 50.21%, 2D 32.69%.
M-S2 eligible: **False**. Failed gates: wrist_below_historical_by_10pct, tips_below_historical_by_10pct, real_dev_last_three_evaluations_improving, no_severe_domain_imbalance.
Remaining severe domain regression: rhd/visible_2d_epe_px.
Training collapse records: `{"affected_updates": 14, "first_update": 302, "last_update": 767, "maximum_batch_rate": 0.03333333507180214, "raw_evidence": "steps.jsonl; retained independently of later dev recovery"}`.

- last.pt SHA256: `4416633ceec579725fa39678a64fbd95b2ad0f034e1d52e97ad857cbc353ca09`
- best_dev.pt SHA256: `74e2e2a287cc4fd4e83c7f98414ce46a618fbb474e56dff4bcd285f0e4296795`

The comparison and descriptive paired 2D intervals are in `artifacts/m_s1_final_comparison_20261022/comparison.md` and `comparison.json`. Per-run reports retain all domain metrics, train/dev curves, masks, data inventories, exposure counts, early failures and geometry intervals.
The dev set contains one independent DexYCB subject; the sequence intervals do not establish cross-subject generalization. The correction uses the same dev for selection, so its comparison is descriptive rather than a final test.
Stopped after this single M-S1 correction. No third run, M-S2, E3, Final-E4 inference, KD, B2 tuning, Dyn-HaMR integration or retargeting was run.
