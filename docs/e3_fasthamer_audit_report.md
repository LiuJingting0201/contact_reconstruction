# E3 checkpoint audit and retained ViT+B2 baseline

Formal E3 is **blocked: no qualified author-published Fast-HaMeR ResNet
hand-reconstruction checkpoint was found in the audited official repository,
README, download scripts, release or tag listings**. No ResNet inference,
training, E3 optimization or E3 latency benchmark was run. The compensation
claim and replacement decision remain untested. ViT+B2 remains the retargeting
baseline. Existing Real-E2, pose ablation and B2 configuration were not changed.

## Author source and checkpoint gate

Audit date: 2026-10-20. Official source:
https://github.com/hunainahmedj/Fast-HaMeR .
Current `main` and the downloaded source resolve to
`009821c80e9bfa4e111c6b9b641408126c53d03f`.
The recursive GitHub tree is complete; 128 copied text/source blobs match their
Git object hashes, with zero mismatches. Public releases and tags are empty.
Raw API responses, source snapshot, HTTP evidence, URL scan and integrity results
are retained in `artifacts/e3_fasthamer_audit_20261020_01/`.

| Evidence | Finding |
|---|---|
| README lines 98, 108, 111, 132 | Teacher is default HaMeR; student examples require user-supplied `path/to/student.ckpt` or locally trained `logs/.../last.ckpt` |
| `fetch_demo_data.sh` | Google Drive ID `1mv7CUAnm73oKsEEG1xE3xH2C_oqcFSzT`; alternate `https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz` |
| `hamer/models/__init__.py` | The same original HaMeR archive; default `_DATA/hamer_ckpts/checkpoints/hamer.ckpt` |
| Download training script | Training datasets, not trained student weights |
| Official tree / releases / tags | No student `.ckpt`/`.pth`/`.pt` asset or release download |
| Public issue listing | Requests titled “Request for pre-trained checkpoints for lightweight models” and “Release Fast-HaMeR student models on Hugging Face” |
| Issue comments / Hugging Face search | No author checkpoint reply; `Fast-HaMeR` model-name search returns an empty list |

The demo archive is the original ViT teacher, not a ResNet student. It was not
downloaded again. No author ResNet checkpoint URL was identified, so there is no
qualified ResNet checkpoint hash or checkpoint training provenance to report.
This is an evidence-bounded availability conclusion, not a claim about private
author checkpoints or every third-party host.

ImageNet ResNet weights, random MANO-head initialization and the project's
128-image RHD pilot are **not** Fast-HaMeR trained hand-reconstruction checkpoints.
`load_efficient_hamer` defaults to the teacher path and uses `strict=False`;
successful loading alone cannot establish a valid student model. A future
checkpoint gate must validate architecture/config, all required keys, training
provenance and hash before inference.

## Frontend contract from source

The `resnet_50*.yaml` family specifies torchvision **ResNet-50**, truncated before
global pooling/classification, 2048 channels with 8×8 features at 256×256 input.
The backbone factory initializes `ResNet50_Weights.IMAGENET1K_V2`. ResNet-101 and
wide-ResNet variants are also present; their existence does not resolve the
requested trained ResNet-50 checkpoint.

The repository describes knowledge distillation, but includes both supervised
`resnet_50.yaml` and feature/output/full-KD configs. The supervised config uses
2D/3D joints, global orientation, articulated pose, betas and adversarial losses.
KD configs add teacher losses. No published checkpoint is available to classify
as supervised or distilled; checkpoint training type is **unavailable**.

Inference uses RGB, 256×256 crops and ImageNet mean `[.485,.456,.406]`,
std `[.229,.224,.225]`. `ViTDetDataset` forms a square expanded bounding box;
its class default expansion is 2.5, while demo CLI default is 2.0. Optional
`BBOX_SHAPE` affects crop expansion. Left hands are mirrored to a right-hand
canonical crop, with camera-x and mesh-x corrected on export. The present
32-sample baseline contains right hands only. A future E3 must preserve the
existing detector boxes and transforms rather than adopting demo defaults.

The head is a six-layer, eight-head cross-attention transformer decoder
(2048 context channels), predicting residual 6D MANO rotations, 10 betas and
three camera parameters, then converting to rotation matrices. MANO supplies
21 joints. The head predicts global orientation and 15 articulated rotations.

Camera output is weak-perspective `(s, tx, ty)`, converted to crop-space
perspective translation `(tx,ty,2*f/(image_size*s+1e-9))`, with nominal crop focal
5000 px from defaults. Demo full-image camera conversion uses scaled nominal
focal length and image-center principal point; it is not calibrated stereo K.
The fixed project real-K initialization instead uses calibrated crop K, with
the geometric mean of anisotropic crop fx/fy for weak-camera depth. Dual-focal
pseudo observations are camera interpretations of each view's own prediction.
These conventions must be preserved if a qualified student checkpoint arrives.

## Fixed E3 matrix: available cached references

All values below are means over the same 32 existing samples, without selection.
A is **cached B0 / E2-0 real-K root initialization**, using the left-view MANO
state shared by both views. B is the cached frozen B2 output. Raw does not mean
the nominal-focal demo-camera output or an independently aggregated right-view
3D estimate. No new frontend result is substituted into these references.

| Metric | E3-A ViT raw reference | E3-B ViT+B2 reference | E3-C ResNet raw | E3-D ResNet+B2 |
|---|---:|---:|---|---|
| Absolute MPJPE, mm | 41.911 | 8.125 | blocked | blocked |
| Wrist-relative MPJPE, mm | 6.762 | 6.762 | blocked | blocked |
| Rotation-aligned diagnostic MPJPE, mm | 4.038 | 4.038 | blocked | blocked |
| Left wrist depth error, mm | 39.355 | 6.476 | blocked | blocked |
| Right wrist depth error, mm | 29.896 | 6.685 | blocked | blocked |
| Wrist position error, mm | 41.974 | 8.847 | blocked | blocked |
| Fingertip absolute error, mm | 41.721 | 9.660 | blocked | blocked |
| Fingertip wrist-relative error, mm | 9.326 | 9.326 | blocked | blocked |
| Thumb bone-direction angle, degrees | 5.688 | 5.688 | blocked | blocked |
| Index bone-direction angle, degrees | 6.061 | 6.061 | blocked | blocked |
| Middle bone-direction angle, degrees | 6.258 | 6.258 | blocked | blocked |
| Ring bone-direction angle, degrees | 6.988 | 6.988 | blocked | blocked |
| Little bone-direction angle, degrees | 8.777 | 8.777 | blocked | blocked |
| Left GT-2D EPE, px | 10.026 | 3.656 | blocked | blocked |
| Right GT-2D EPE, px | 20.532 | 3.471 | blocked | blocked |
| Pseudo reprojection residual, px | 14.236 | 3.555 | blocked | blocked |
| Numerical / depth failures | 0 / 0 | 0 / 0 | unavailable | unavailable |
| Quality failures | 18/32 | 1/32 | unavailable | unavailable |

Authority: `dex_pose_preserving_20261020_01/independent_metrics_final.json`.
The earlier conversational summary's raw 24.71 mm is not the value in this
authoritative B0 aggregate and is not used here. No old metrics were overwritten.
Machine-readable sample metrics and mean/median/P95 are copied into the export's
`baseline_metrics.json`, preserving the exact original definitions.

ViT teacher SHA256:
`e5cc06f294d88a92dee24e603480aab04de532b49f0e08200804ee7d90e16f53`.
Frozen B2: calibrated left/right stereo, normalized Huber reprojection at 3 px,
translation only, seed 42, 800 steps, translation LR .001, decay at step 600
by .1, minimum joint depth .01 m. All global/articulated rotations and shape
remain frozen. Exact full configuration is copied without modification.

## Timing and replacement decision

Detector, crop, network, MANO, B2, frontend, end-to-end latency and peak GPU memory
are **unavailable under the required E3 benchmark protocol**. Historical B2
32-sample runtime/memory values are preserved only as audit evidence; they do not
meet 20 warm-ups plus 100 timed iterations, batch 1, synchronized CUDA, separated
stages, mean/P50/P95 and consistent resident-model memory scopes. They are not
reported as E3 speed or memory claims.

Raw degradation, optimized degradation, compensation ratio, frontend speedup,
end-to-end speedup and memory reduction are all **unavailable**, because C/D
were not executed. There is no measured basis to approve ResNet as a replacement.
For a future run, compensation ratio must additionally be marked undefined if
raw degradation is zero.

The replacement gates remain: absolute increase ≤max(2 mm, 15% of ViT+B2),
wrist-relative increase ≤2 mm, fingertip wrist-relative increase ≤3 mm,
quality failures ≤1/32, frontend speedup ≥1.5× and net end-to-end acceleration.
With this baseline, the absolute limit is 10.1254 mm; wrist-relative limit is
8.7619 mm and fingertip wrist-relative limit is 12.3260 mm.

B2 can correct root translation and absolute depth; it cannot alter orientation,
internal finger articulation or shape. Its preservation of wrist-local geometry
does not demonstrate that it can compensate a student's finger-pose errors.

## Checkpoint gap and reasonable training-data requirement

The immediate missing deliverable is an author-trained ResNet checkpoint with its
model config, training recipe/data/split provenance, license and SHA256. Acquiring
that asset is preferable to inventing a substitute benchmark.

If training is later authorized, use a licensed, diverse hand-reconstruction
training mixture with real 2D annotations, metric 3D and MANO supervision where
available, and frozen ViT teacher outputs for any declared distillation. The
repository's mix includes FreiHAND, InterHand2.6M, MTC, RHD, COCO-WholeBody,
HALPE, MPII NZSL, HO3D, H2O3D and DexYCB. Respect each source's permissions.
Real calibrated multiview data is particularly useful for position/depth,
while articulation supervision must cover occluded fingers, interactions,
both hands, subjects and camera domains.

A reasonable planning scale is tens of thousands to hundreds of thousands of
distinct annotated hand crops, with subject/sequence-disjoint validation and
learning-curve evidence; this is an engineering budget recommendation, not a
proven minimum or an assertion of the authors' exact training size. The source
config has a one-million-step, batch-eight budget, which describes repeated
optimization samples rather than one million unique images. Protect these
32 benchmark IDs and their sequences from training leakage. Neither the
128-image RHD pilot nor these 32 benchmark frames support a formal trained
replacement conclusion. No such training was started.

## Retargeting baseline and verification

Authoritative export: `artifacts/e3_vit_b2_retargeting_20261020_03/`.
It contains all 32 samples in JSON/NPZ, left-camera MANO joints in meters,
wrist-local joints, global orientation, articulated rotations, shape,
timestamps/identities, quality flags and camera calibration metadata.
See `docs/e3_retargeting_schema.md`.

Confidence is explicitly unavailable rather than invented. Non-GT validity is
finite/positive-depth only. The one GT-assisted quality failure remains as a
separate evaluation-only annotation; no sample is filtered. Time is nominal
frame/30, not measured capture time. Wrist-local axes remain camera axes.

Verification passed for ordered IDs, exact frozen rotation/shape equality,
unchanged solver calibration, exact cached-B2 export, JSON/NPZ round-trip,
finite/positive depth, copied-evidence hashes and unchanged source files.
Prior independently verified checkpoint reload evidence is reused. Inference
and optimization calls are both zero. Intermediate export directories `_01`
and `_02` retain failed attempts; `_03` is authoritative.

Stop point reached: checkpoint audit and blocked-E3 report complete, ViT+B2
baseline retained. No YOLO training, large-model training, TensorRT or VLA work.
