# Dyn-HaMR Hand Reconstruction Improvement Experiment Plan

> **Superseded / 已归档。** 当前执行规范是 [会议方案复现与工程落地 v1.1](dyn_hamr_reproduction_protocol.md)，原始全文备份在 [reference](reference/improvement_plan_original.md)。本页保留历史内容，不再用于执行；尤其下文“无 KD 等于随机初始化”、默认能力、焦距定义与运行状态已由新规范纠正。实际进度以 [实验记录](dyn_hamr_experiment_record.md) 为准。

## 1. Document Information

- Project: stereo 3D hand reconstruction based on Dyn-HaMR
- Meeting source: `文字记录：手姿3D重建技术落地专项会 2026年10月14日`
- Target repository: <https://github.com/ZhengdiYu/Dyn-HaMR>
- Related lightweight model: <https://github.com/hunainahmedj/Fast-HaMeR>
- Document status: experiment design, pending baseline reproduction
- Main language: Chinese, with model and metric names retained in English

## 2. Purpose

This document converts all technical proposals in the meeting into reproducible experiments. It covers:

1. The focal-length inconsistency in HaMeR/Dyn-HaMR.
2. Dual-focal initialization and 2D pseudo-label generation.
3. Stereo reprojection optimization.
4. Lightweight backbone training with and without distillation.
5. YOLO hand detector retraining.
6. Tracking cleanup and temporal filtering.
7. GPU-resident preprocessing and pipeline parallelism.
8. FP16, TensorRT, INT8, and optimizer experiments.

The main objective is not only to obtain visually plausible hand meshes, but to improve metric depth, stereo consistency, temporal stability, and end-to-end throughput simultaneously.

## 3. Meeting Claims and Interpretation

| ID | Meeting statement | Engineering interpretation | Verification status |
|---|---|---|---|
| C1 | HaMeR assumes focal length 5000 | HaMeR predicts a weak-perspective camera and uses a nominal focal length when converting it to perspective translation | Confirmed in code |
| C2 | Using real focal length improves depth but hurts 2D alignment | Real intrinsics produce a more plausible camera distance, but expose perspective errors in the predicted MANO pose | To be measured |
| C3 | Using focal length 5000 gives accurate 2D but incorrect depth | Large focal length keeps the projection close to weak perspective while scaling `tz` to an unrealistic distance | To be measured |
| C4 | Generate both results at inference time | Produce a fixed-focal 2D observation and a real-focal 3D initialization from the same image | Core experiment |
| C5 | Use fixed-focal 2D as ground truth | Treat the 5000-focal projection as a pseudo-label, not real annotated ground truth | Core experiment |
| C6 | Use real-focal hand as initialization | Initialize MANO pose, root orientation, shape, and especially camera-space translation using calibrated intrinsics | Core experiment |
| C7 | Optimize with reprojection loss | Reproject the initialized 3D hand through the real camera and optimize it toward the pseudo-2D observations | Confirmed available in Dyn-HaMR |
| C8 | Add stereo reprojection loss | Optimize one shared 3D hand against synchronized left and right camera observations | Core experiment |
| C9 | The repository contains incorrect camera parameters | Current code has inconsistent focal defaults and overrides | Confirmed in code |
| C10 | Use datasets recommended by Fast-HaMeR | Reuse its training/evaluation data manifests where licenses permit, then add product-domain stereo data | Pending data inventory |
| C11 | A small ResNet can be trained without distillation | Train lightweight backbones from scratch as a first-class baseline | Core experiment |
| C12 | Distillation improves only a few points | Compare no distillation, output distillation, feature distillation, and hybrid distillation | To be measured |
| C13 | Retraining YOLO is important | Fine-tune the hand detector on product-domain images, including difficult left/right and occlusion cases | Core experiment |
| C14 | Dirty tracks need filtering and smoothing | Add detection rejection, handedness correction, gap handling, and temporal filtering | Core experiment |
| C15 | Remove CPU/GPU transfers and cache overhead | Keep decode, crop, preprocessing, inference, and intermediate tensors on GPU | Performance experiment |
| C16 | Decode directly to GPU with NVDEC | Replace CPU video decode and upload with GPU decode | Performance experiment |
| C17 | Deeply pipeline preprocessing and inference | Overlap decode, detection, crop/blur, regression, and optimization using streams and bounded queues | Performance experiment |
| C18 | Use half precision | Evaluate FP16/BF16 where numerically safe | Performance experiment |
| C19 | Use TensorRT INT8 | The transcript phrase "SRT init 8" is interpreted as TensorRT INT8 | Performance experiment |
| C20 | Change optimizer | Benchmark optimization schedules instead of assuming the repository default is optimal | Performance and convergence experiment |
| C21 | Existing optimized implementation was about 100x faster | Treat this as a claim to reproduce, not as an established result | To be measured |
| C22 | Small model only needs accurate 2D projection | Test whether downstream geometric optimization can recover 3D quality from a weaker but 2D-accurate initializer | Main model hypothesis |

## 4. Code Audit Findings

### 4.1 HaMeR camera conversion

HaMeR predicts weak-perspective camera parameters:

\[
\hat{c}=(s,t_x,t_y)
\]

The Dyn-HaMR HaMeR branch converts them to full-frame camera translation using:

\[
t_z=\frac{2f}{b\,s+\epsilon}
\]

\[
T_x=t_x+\frac{t_z}{f}(c_x-W/2), \qquad
T_y=t_y+\frac{t_z}{f}(c_y-H/2)
\]

where:

- \(f\): focal length in pixels;
- \(b\): hand bounding-box size in pixels;
- \((c_x,c_y)\): bounding-box center;
- \((W,H)\): full image size.

Relevant code:

- `ZhengdiYu/hamer`, branch `dynamic-hamer`, `run.py:1498-1515`
- HaMeR default: `hamer/configs/__init__.py:53`, `FOCAL_LENGTH = 5000`
- Dyn-HaMR branch override: `hamer/models/__init__.py:45`, `FOCAL_LENGTH = 500.0`

The `5000` versus `500.0` mismatch must be resolved before any accuracy or runtime comparison.

### 4.2 Current 2D observations

The Dyn-HaMR branch exports HaMeR's projected 2D joints as `extra_data`, writes them as keypoint JSON files, and later loads them as `joints2d`. Therefore, the downstream optimization is not necessarily supervised by an independent 2D detector.

The current dataset loader also forces all loaded confidence values to `1.0`. This can make hallucinated or occluded joints contribute as strongly as reliable joints.

### 4.3 Existing optimization support

Dyn-HaMR already provides:

- Perspective reprojection through supplied camera intrinsics/extrinsics.
- Robust 2D joint loss.
- 3D temporal smoothness.
- Pose and shape priors.
- Motion prior.
- Bone-length and biomechanical constraints.
- Penetration loss.
- World-scale optimization.

The primary algorithmic change should therefore focus on constructing consistent observations and initialization, then extending the loss to calibrated stereo cameras.

## 5. Main Technical Hypothesis

The original pipeline asks one prediction to satisfy two conflicting requirements:

1. Preserve the weak-perspective behavior learned by HaMeR.
2. Represent the hand at the correct metric distance under the real camera.

The proposed method separates these responsibilities:

- Fixed-focal branch: supplies stable 2D pose evidence.
- Real-focal branch: supplies plausible depth and translation initialization.
- Stereo optimization: reconciles both in a calibrated, physically meaningful camera model.

The expected improvement comes from reducing initialization error and making depth observable through stereo parallax.

## 6. Proposed Method

### 6.1 Inputs

For each synchronized timestamp \(t\) and view \(v \in \{L,R\}\):

- RGB image \(I_{t,v}\)
- Camera intrinsics \(K_v\)
- Lens distortion parameters \(D_v\)
- Camera extrinsics \(T_{v\rightarrow rig}\)
- Stereo synchronization metadata

Required calibration quality:

- Intrinsics and distortion estimated for the actual capture resolution.
- Stereo extrinsics fixed in a shared rig coordinate frame.
- Rectification error recorded in pixels.
- Timestamp mismatch measured and reported.

### 6.2 Hand detection and tracking

1. Decode frames.
2. Run the retrained YOLO hand detector.
3. Assign left/right handedness.
4. Associate detections temporally.
5. Associate left-view and right-view tracks.
6. Reject impossible overlaps and duplicate detections.
7. Interpolate only short gaps.
8. Preserve confidence and provenance for every observation.

Each observation must record whether it is:

- directly detected;
- corrected by handedness logic;
- interpolated;
- copied by patience logic;
- rejected as a hallucination.

### 6.3 HaMeR inference and dual-focal outputs

Run the hand regressor to obtain:

- MANO pose \(\theta\);
- MANO shape \(\beta\);
- root orientation \(\phi\);
- weak-perspective camera \((s,t_x,t_y)\);
- root-relative joints and vertices.

Then construct two camera interpretations.

#### Branch A: virtual focal length

- Set \(f_{virtual}=5000\), after normalizing for crop and image resolution.
- Compute \(T_{virtual}\).
- Project MANO joints to obtain \(u^{pseudo}_{t,v}\).
- Use these joints as 2D pseudo-labels.

#### Branch B: calibrated focal length

- Use \(K_v\) and the real focal length.
- Compute \(T_{real}\).
- Use \((\theta,\beta,\phi,T_{real})\) as the 3D initialization.

Important implementation observation:

In the current HaMeR code, focal length is not an input to the ViT backbone or MANO decoder. It is used after regression to compute camera translation and projected 2D joints. Therefore, the first implementation should use:

> one network forward pass plus two camera post-processing branches.

A true two-forward implementation is needed only for a separately trained focal-conditioned model.

### 6.4 Stereo shared-state optimization

At each timestamp, optimize one world-space hand state \(Q_t\), shared by both cameras:

\[
Q_t=\{\theta_t,\beta,\phi_t,\tau_t\}
\]

For view \(v\):

\[
\hat{u}_{t,v} =
\pi\left(K_v,T_{rig\rightarrow v},J(Q_t)\right)
\]

The stereo reprojection objective is:

\[
\mathcal L_{stereo} =
\sum_t\sum_{v\in\{L,R\}}\sum_j
w_{t,v,j}
\rho\left(
\left\|\hat{u}_{t,v,j}-u^{pseudo}_{t,v,j}\right\|_2^2
\right)
\]

Recommended full objective:

\[
\mathcal L =
\lambda_{2d}\mathcal L_{stereo}
+\lambda_{pose}\mathcal L_{pose}
+\lambda_{shape}\mathcal L_{shape}
+\lambda_{temp}\mathcal L_{temporal}
+\lambda_{bone}\mathcal L_{bone}
+\lambda_{bio}\mathcal L_{bio}
+\lambda_{pen}\mathcal L_{penetration}
+\lambda_{depth}\mathcal L_{depth}
\]

Confidence \(w_{t,v,j}\) must combine:

- detector confidence;
- pseudo-keypoint confidence;
- visibility/occlusion status;
- track provenance;
- left-right consistency;
- temporal outlier score.

Do not force every confidence value to `1.0`.

### 6.5 Optional focal-conditioned model

The meeting mentions a prior approach that:

- augments focal length during training;
- parameterizes focal length;
- feeds focal length into the model.

This should be treated as an independent experiment, not mixed into the initial dual-focal implementation.

Candidate conditioning variables:

- normalized \(f_x/W\), \(f_y/H\);
- horizontal and vertical field of view;
- crop-to-full-image transform;
- principal-point offset;
- bounding-box scale.

The focal embedding can be concatenated with the MANO decoder query or injected through FiLM conditioning.

## 7. Experiment Questions

### Q1. Is the fixed-focal result actually more accurate in 2D?

Compare fixed focal values `500`, `1000`, `2000`, `5000`, and calibrated focal length.

Measure:

- 2D keypoint error against manually checked or independent detector annotations;
- silhouette IoU;
- per-joint error by depth variation;
- error versus field of view and hand-to-camera distance.

### Q2. Does real focal length improve metric depth?

Measure root depth and camera-space joint error against:

- calibrated stereo triangulation;
- RGB-D ground truth, if available;
- multi-view fitted MANO ground truth;
- mocap ground truth, if available.

### Q3. Does dual-focal optimization outperform either branch alone?

Compare:

- fixed-focal-only;
- real-focal-only;
- fixed 2D plus real 3D initialization;
- fixed 2D plus real 3D initialization with confidence weighting;
- independent 2D detector plus real 3D initialization.

### Q4. How much does stereo contribute?

Compare:

- left monocular;
- right monocular;
- best-view selection;
- stereo reprojection;
- stereo reprojection plus triangulated wrist/root initialization.

### Q5. Can a lightweight model preserve final optimized 3D quality?

Compare ViT-H with ResNet, ConvNeXt, MobileNet, and MobileViT students before and after geometric optimization.

This experiment must report both:

- raw model accuracy;
- final pipeline accuracy after stereo optimization.

A weaker raw 3D model may still be acceptable if final optimized 3D remains stable.

## 8. Experiment Matrix

### 8.1 Phase A: baseline and focal-length audit

| Exp | Regressor | Focal for 2D | Focal for initialization | Views | Purpose |
|---|---|---:|---:|---:|---|
| A0 | Current repository | Current config | Current config | Mono | Reproduce current behavior |
| A1 | HaMeR | 5000 | 5000 | Mono | Weak-perspective baseline |
| A2 | HaMeR | Real | Real | Mono | Direct calibrated-camera baseline |
| A3 | HaMeR | 5000 | Real | Mono | Isolate dual-focal improvement |
| A4 | HaMeR | Independent 2D detector | Real | Mono | Check pseudo-label bias |
| A5 | HaMeR | 5000 | Real | Stereo | Main proposed geometry method |
| A6 | HaMeR | Independent 2D detector | Real | Stereo | Stereo detector-supervised control |
| A7 | Focal-conditioned HaMeR | Real | Real | Stereo | Learned-camera alternative |

### 8.2 Phase B: loss ablation

Start from A5 and remove one component at a time:

| Exp | Removed or changed component |
|---|---|
| B0 | Full objective |
| B1 | No temporal smoothness |
| B2 | No pose prior |
| B3 | No bone-length constraint |
| B4 | No biomechanical constraint |
| B5 | No penetration loss |
| B6 | No depth constraint |
| B7 | All observation confidence set to 1 |
| B8 | No robust loss |
| B9 | Left-view reprojection only |
| B10 | Right-view reprojection only |

### 8.3 Phase C: detector and track cleanup

| Exp | Detector | Cleanup |
|---|---|---|
| C0 | Public YOLO checkpoint | None |
| C1 | Public YOLO checkpoint | Current repository heuristics |
| C2 | Retrained YOLO | None |
| C3 | Retrained YOLO | Overlap and hallucination rejection |
| C4 | Retrained YOLO | C3 plus handedness correction |
| C5 | Retrained YOLO | C4 plus gap interpolation |
| C6 | Retrained YOLO | C5 plus temporal filtering |

### 8.4 Phase D: lightweight model

For each student backbone:

| Exp | Training strategy |
|---|---|
| D0 | Train from scratch, standard HaMeR losses |
| D1 | Output-level distillation |
| D2 | Feature-level distillation |
| D3 | Output plus feature distillation |
| D4 | Train from scratch with stronger 2D loss |
| D5 | Train from scratch with focal augmentation |

Candidate backbones:

- ResNet-18
- ResNet-50
- ConvNeXt-Tiny
- MobileNetV3
- MobileViT

The first deployment candidate should be selected using final pipeline accuracy per millisecond, not raw model accuracy alone.

### 8.5 Phase E: system optimization

Apply changes cumulatively and also profile them independently:

| Exp | Change |
|---|---|
| E0 | Original repository pipeline |
| E1 | Eliminate redundant CPU-to-GPU and GPU-to-CPU transfers |
| E2 | Reuse device buffers and pinned host buffers |
| E3 | NVDEC GPU decode |
| E4 | GPU crop, resize, blur, normalize, and color conversion |
| E5 | Overlap decode, detection, preprocessing, and regression |
| E6 | FP16/BF16 inference |
| E7 | TensorRT FP16 |
| E8 | TensorRT INT8 with calibration |
| E9 | Lightweight backbone plus TensorRT |
| E10 | Optimized solver and reduced iteration schedule |

## 9. Dataset Plan

### 9.1 Public data

Use the datasets and preprocessing recipes listed by Fast-HaMeR after confirming:

- download availability;
- annotation format;
- commercial-use constraints;
- MANO licensing;
- left/right-hand labels;
- camera intrinsics availability.

Also consider the datasets used by Dyn-HaMR:

- H2O
- EgoDexter
- FPHA
- HOI4D
- InterHand2.6M

Public datasets should cover:

- isolated hands;
- two-hand interaction;
- hand-object interaction;
- severe occlusion;
- egocentric views;
- wide field of view;
- dynamic cameras.

### 9.2 Product-domain data

Create a calibrated stereo evaluation set with:

- synchronized left/right frames;
- diverse distances and fields of view;
- fast motion and motion blur;
- self-occlusion and two-hand interaction;
- hands entering and leaving the frame;
- difficult handedness cases;
- representative lighting and skin tones.

Split by subject and scene, not by frame, to avoid leakage.

### 9.3 Ground truth hierarchy

Use the strongest available label per sample:

1. Multi-camera mocap or marker-based ground truth.
2. Calibrated multi-view MANO fitting reviewed manually.
3. RGB-D depth plus manual 2D keypoints.
4. Stereo triangulated joints.
5. Independent high-quality 2D annotations.

Pseudo-labels must never be reported as ground truth metrics.

## 10. Metrics

### 10.1 3D geometry

- MPJPE in millimeters.
- PA-MPJPE.
- MPVPE and PA-MPVPE.
- Wrist/root translation error.
- Absolute depth error.
- Relative depth error between two hands.
- Percentage of joints within 5, 10, 20, and 30 mm.
- Stereo triangulation consistency.
- Bone-length variance.
- Inter-hand penetration depth and frequency.

### 10.2 2D alignment

- Mean reprojection error in pixels.
- Normalized reprojection error by hand bounding-box diagonal.
- PCK and AUC.
- Silhouette IoU where segmentation is available.
- Per-view and worst-view reprojection error.

### 10.3 Temporal quality

- Joint velocity error.
- Joint acceleration error.
- Jitter energy.
- Root trajectory drift.
- Track fragmentation count.
- Left/right identity switch count.
- Missing-frame recovery accuracy.

### 10.4 Detection

- Box AP50 and AP50:95.
- Recall at product operating threshold.
- Left/right classification accuracy.
- False duplicate rate.
- Miss rate under occlusion and blur.

### 10.5 Performance

- End-to-end FPS.
- Per-stage latency: p50, p90, and p99.
- Throughput at batch sizes 1, 2, 4, 8, and 16.
- Peak GPU memory.
- GPU utilization.
- CPU utilization.
- Host-to-device and device-to-host transfer count and bytes.
- Decode queue, preprocessing queue, and inference queue occupancy.
- TensorRT engine build and calibration time.

All latency measurements must include warm-up and use CUDA events for GPU stages.

## 11. Runtime Architecture

Recommended staged pipeline:

1. NVDEC decode to GPU surfaces.
2. GPU color conversion and resize.
3. YOLO detection.
4. GPU crop and crop batching.
5. GPU blur/normalization/augmentation as required.
6. HaMeR or lightweight regressor inference.
7. Dual-focal camera post-processing.
8. Stereo association.
9. Batched geometric optimization.
10. GPU rendering or asynchronous result export.

Use:

- bounded queues to control memory;
- reusable tensor pools;
- dedicated CUDA streams;
- events for inter-stage dependencies;
- asynchronous output writing;
- no per-frame `.cpu().numpy()` conversion in the hot path.

The current repository exports many per-frame JSON and NumPy objects. Keep this debug path for validation, but add a binary batched path for production benchmarking.

## 12. Optimizer Experiments

The repository uses an LBFGS-based multi-stage optimizer. Evaluate:

- LBFGS baseline.
- Adam followed by LBFGS refinement.
- AdamW with cosine or plateau scheduling.
- Gauss-Newton or Levenberg-Marquardt for reprojection-dominant stages.
- Fewer iterations with early stopping.
- Fixed pose with translation-only initialization stage.
- Root pose stage followed by articulation stage.
- Sliding-window versus full-sequence optimization.

Convergence must be compared using:

- final objective value;
- final 2D and 3D error;
- iterations to threshold;
- wall-clock latency;
- failure rate and NaN rate.

## 13. Required Ablation Reporting

Every result table must report:

- exact Git commit;
- checkpoint hash;
- focal-length convention;
- crop and full-frame coordinate conventions;
- camera calibration version;
- input resolution;
- precision mode;
- optimizer and iteration count;
- detector checkpoint;
- dataset split;
- hardware and software versions.

The focal value alone is insufficient. Record whether it is expressed in:

- crop pixels;
- full-image pixels;
- normalized image coordinates;
- horizontal or vertical focal length.

## 14. Expected Outcomes

### Dual-focal plus stereo

Expected:

- lower metric depth error than fixed-focal-only inference;
- lower 2D error than real-focal-only inference;
- less root-depth drift than monocular optimization;
- fewer optimization failures due to a better initial state.

### Lightweight model

Expected:

- worse raw 3D accuracy than ViT-H;
- similar 2D projection quality for a sufficiently trained ResNet/ConvNeXt;
- smaller final accuracy gap after stereo optimization;
- substantially lower inference latency and memory.

### Detector retraining

Expected:

- higher recall on product-domain images;
- fewer handedness switches and duplicate tracks;
- fewer invalid frames entering geometric optimization.

### System optimization

Expected:

- lower decode and preprocessing latency;
- fewer synchronization stalls;
- higher sustained GPU utilization;
- TensorRT speedup limited by the remaining non-model stages.

The meeting's reported "100x" improvement must be decomposed by stage and reproduced against a precisely defined baseline.

## 15. Risks and Controls

### Pseudo-label confirmation bias

The 5000-focal 2D prediction can be confidently wrong.

Controls:

- compare against independent 2D detections;
- preserve per-joint confidence;
- use robust loss;
- reject cross-view inconsistent pseudo-labels;
- manually audit a fixed failure subset.

### Stereo synchronization error

Fast hand motion makes small timestamp offsets look like geometric inconsistency.

Controls:

- hardware synchronization where possible;
- estimate temporal offset;
- exclude high-motion unsynchronized samples;
- report performance by motion speed.

### Calibration error

Bad intrinsics or extrinsics can make stereo optimization worse than monocular fitting.

Controls:

- report rectification residual;
- validate calibration with static targets;
- version calibration files;
- keep camera parameters fixed in the first experiment.

### Incorrect focal units

Crop-space and full-frame focal lengths are easy to mix.

Controls:

- centralize camera conversion in one tested module;
- add round-trip projection unit tests;
- visualize axes, principal points, and projected calibration targets;
- prohibit hidden focal overrides.

### Over-smoothing

Filtering may reduce jitter while erasing fast finger motion.

Controls:

- measure both jitter and motion amplitude;
- evaluate by motion speed;
- apply confidence-aware filtering;
- retain an unsmoothed output for geometry evaluation.

### INT8 accuracy loss

Quantization may damage fine-grained pose regression.

Controls:

- use representative calibration data;
- keep sensitive heads in FP16;
- report per-joint degradation;
- compare raw and post-optimization accuracy.

## 16. Execution Order

### Milestone 0: reproducibility

- Pin repository and submodule commits.
- Reproduce one official Dyn-HaMR example.
- Resolve `5000` versus `500.0`.
- Add per-stage timing and camera-parameter logging.

### Milestone 1: geometry proof

- Build a small calibrated stereo subset.
- Run A0-A6.
- Confirm or reject C2-C8.
- Select the 2D observation source.

### Milestone 2: robustness

- Retrain YOLO.
- Run C0-C6.
- Add confidence-aware temporal processing.
- Repeat geometry experiments on difficult sequences.

### Milestone 3: lightweight model

- Train ResNet baseline without distillation.
- Run D0-D5.
- Select the smallest model meeting final pipeline quality.

### Milestone 4: performance

- Profile the unmodified pipeline.
- Implement E1-E10 incrementally.
- Produce an accuracy-latency Pareto curve.

## 17. Initial Acceptance Criteria

Final thresholds should be fixed after baseline measurement. The initial decision rules are:

- Dual-focal stereo must reduce absolute depth error without materially degrading 2D reprojection.
- Stereo must outperform both monocular views on root-depth and trajectory metrics.
- Detector retraining must improve recall without an unacceptable increase in duplicate detections.
- A lightweight model is accepted only if final post-optimization 3D quality remains within the agreed product tolerance.
- Every performance optimization must preserve the selected geometry-quality threshold.
- TensorRT INT8 is accepted only if its final post-optimization degradation is smaller than its deployment speed benefit justifies.

## 18. Deliverables

- Baseline reproduction report.
- Camera convention and focal-length audit.
- Dual-focal implementation.
- Stereo reprojection optimizer.
- Public and product-domain dataset manifests.
- YOLO retraining configuration and checkpoint.
- Lightweight model training configurations and checkpoints.
- End-to-end profiler traces.
- Accuracy, stability, and latency ablation tables.
- Failure-case gallery with original frames, both focal branches, and final stereo result.
- Deployment recommendation with one accuracy-first and one speed-first configuration.

## 19. Open Questions

1. Does the production setup provide calibrated stereo intrinsics, distortion, extrinsics, and hardware synchronization?
2. What is the required metric depth tolerance?
3. Is final output required online, near-real-time, or offline?
4. What GPU architecture is the deployment target?
5. Is the phrase transcribed as "Wheeler" referring to WiLoR or another focal-conditioned method?
6. Does "三五厘米级" refer to depth error, global translation error, or another metric?
7. Which optimizer change was used in the prior implementation?
8. Which exact ResNet variant and training dataset mixture produced the reported result?
9. What baseline and hardware were used for the reported 100x speedup?

These questions do not block Milestone 0 or the focal-length audit, but they must be resolved before claiming final accuracy or runtime gains.
