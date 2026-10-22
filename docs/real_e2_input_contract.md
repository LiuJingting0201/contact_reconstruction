# Real E2 input contract and executable entry

Latest status on 2026-10-20: **official DexYCB real matrix and ±1-frame sensitivity completed**.
The historical blocked contract below is retained for provenance and is superseded
by protocol §22 and the following changes. Do not request user camera collection
or resume unofficial mirror searches.

## Official DexYCB revision

Use official subject-01 (Google Drive ID `1Ehh92wDE3CWAiKG7E9E73HjN2Xk2XfEk`,
12,412,314,463 bytes) and calibration ID `1UAwVKT4Rgb1fLcFoa1o71_-0NtSvvLAQ`.
The normal Drive large-file confirmation succeeded (HTTP200); no permission bypass.
The full archive was downloaded locally, where 263 GiB was available; selected
sequences were copied to the remote GPU host (24 GB available before work).
Complete gzip CRC/EOF and all 5,314 selected subject/calibration files passed
original-archive comparison. Exact sizes and SHA256 are recorded in
`artifacts/dexycb_official_acquisition_20261020_01/verification.json`.

`prepare_dexycb_real_e2.py` installs/uses official toolkit commit
`64551b001d360ad83bc383157a559ec248fb9100`, retaining its exact subset-selection diff.
Successful preparation: `artifacts/dex_real_e2_prepare_20261020_02`.
32 synchronized times from two right-hand sequences passed projection and transform
checks; eight pairs were visually inspected. Detector boxes use pretrained
MediaPipe Hands, not evaluation GT.

The revised manifest accepts:

- `gt_coordinate_system:"per_camera"`, with original `views[v].gt3d` in camera meters.
  Left-camera reference is internal only; no extra world GT is required.
- `distortion.state:"dataset_delivered_pinhole"`,
  `model:"dataset_pinhole"`, `coefficients:null`, and official guide/code evidence.
  No undistortion is applied. Official camera-GT projection must agree with joint_2d.
- `timestamp_kind:"dataset_frame_index_over_fps"` and exact official sequence/frame
  correspondence. Frame/30 is nominal time, not a measured zero inter-camera offset.
  The guide's approximate synchronizer has `slop=0.1`; physical timestamps unavailable.
- E2-5 non-GT epipolar confidence, explicitly a heuristic. E2-6 unavailable because
  verified oracle joint visibility was not found. Do not use validity or segmentation
  as an undocumented occlusion label.

Run with `--config configs/real_e2_dexycb_official_v2.json`. The reviewed manifest is
`artifacts/dex_real_e2_prepare_20261020_02/manifest_reviewed.json`; real matrix run02
corrects an initial MANO batch-dimension failure, preserving run01.

Completed main run: `artifacts/dex_real_e2_matrix_20261020_02` (32 times,
eight available variants; E2-6 unavailable). Completed sensitivity:
`artifacts/dex_real_e2_sync_20261020_01` (30 common times; cached S0 and 60 new
offset states). All 316 saved states pass independent MANO reconstruction.
Sensitivity correspondence checks confirm unchanged synchronous GT/initialization.
See the latest nine-section entry in `dyn_hamr_experiment_record.md` for results,
tradeoffs and limits. Absolute-position benefit passes the descriptive subset gate;
root-relative error versus raw deteriorates, and E3/ResNet remain paused.

## Historical blocked contract (superseded where noted above)

## Current blockers

1. Project and inspected mounted directories contain no complete real synchronized
   RGB pair set with calibration and independent hand GT.
2. DexYCB official calibration was retrieved (14,464 bytes); an individual RGB was
   retrieved and decoded at 640×480. The annotation-only subject-06 mirror
   `Louis0411/dexycb_mesh`, pinned revision
   `18376c6f7ab2f1cda6f5d316326a67f886171514`, returns **HTTP 401** and the literal
   body `Invalid username or password.`. API and binary range request both fail.
   No permission bypass was attempted. Prior public metadata does not establish
   current download access.
3. The downloaded DexYCB color intrinsics provide fx/fy/ppx/ppy but **no distortion
   coefficients or explicit already-undistorted statement**. The official toolkit
   uses pinhole projection, but that alone does not prove zero lens distortion.
   Do not invent zero coefficients.
4. DexYCB's official extrinsics are **camera-to-world** (camera to master camera),
   as confirmed by official `SequenceLoader._deproject_depth_and_filter_points`
   and the maintainer's reply in issue #6. Convert once with a matrix inverse for
   this interface. Per-sequence `meta.yml` must identify the correct calibration;
   selecting a calibration solely from a date is insufficient.
5. UCBProject's pinned RGB mirror works. A 65,536-byte range of its 5.095 GB
   uncompressed subject-06 archive was inspected; its first members are RGB.
   The range support is real, but this probe did not locate independent labels.
   The entire archive was not downloaded. Official subject packages are about
   12 GB compressed; these were not downloaded either.
6. InterHand2.6M remains an unqualified alternative: an independently annotated,
   paired minimal image/annotation subset and explicit distortion handling were
   not established. STB additionally has unresolved license and uses a palm
   center in place of wrist; do not manufacture wrist GT.

Exact retrieval artifacts are in `artifacts/real_e2_audit_20261020_01/data_search/`.
Source references are in `real_e2_data_investigation.md`. Public dataset availability
in general is not denied: the blocker is that **no verified complete input was
obtained in this session**.

## Required package

Provide `manifest.json` and **20–50 synchronized times** of left/right RGB, with
independent hand annotations. Store paths relative to the manifest. Prefer several
sequences, with documented frontal/tilted palm, wrist rotation, open/bent fingers,
occlusion, image center/edge, and interaction if available. Record missing coverage;
never invent a category. Keep each sequence in one split.

All arrays use OpenCV camera axes (+x right, +y down, +z forward), pixel units for
2D, meters for 3D, and a declared world frame. Transform convention:
`X_camera = R_world_to_camera @ X_world + t_world_to_camera`.
`K` is a 3×3 matrix, `T_world_to_camera` is 4×4. The joint order is wrist, then
thumb/index/middle/ring/little, each MCP/PIP/DIP/tip (the thumb names are labels for
the corresponding MANO chain; supply an explicit dataset mapping).

Top-level JSON fields:

| Field | Required value/type |
| --- | --- |
| `schema`, `kind`, `units` | `"real_e2_v1"`, `"real"`, `"m"` |
| `gt_coordinate_system` | `"world"` |
| `joint_order` | 21 names, exactly `real_e2_data.JOINT_ORDER` |
| `sync_tolerance_s` | Explicit nonnegative tolerance ≤0.005; frame-index-derived timestamps must be labeled as such in evidence |
| `source` | `name`, original URL/revision, `license_url`, `license_allows_experiment:true`, `gt_independent_of_teacher:true`, `gt_method` |
| `visual_sync_review` | `{ "passed": true, "evidence": "path to reviewed overlay/contact-sheet notes" }` after actual review |
| `samples` | 20–50 entries described below |

Each sample:

| Field | Type/meaning |
| --- | --- |
| `sample_id`, `sequence_id`, `sync_id`, `source_original_id` | Stable strings |
| `timestamp_s`, `sync_evidence` | Time and original synchronization provenance |
| `split` | One split per sequence, shared by both views |
| `handedness` | `"left"` or `"right"`, physical hand identity |
| `coverage_tags` | Observed categories, established from RGB; not model accuracy |
| `gt3d`, `gt3d_valid` | Finite 21×3 world meters, 21 booleans (this first matrix requires all 21 GT3D joints) |
| `views` | Exactly two entries ordered left then right |

Each view:

| Field | Type/meaning |
| --- | --- |
| `role`, `image`, `image_sha256` | `"left"`/`"right"`, relative RGB path, SHA256 |
| `timestamp_s`, `sync_id` | Camera's matching time/frame correspondence |
| `resolution_wh`, `K_resolution_wh` | `[width,height]`, both equal decoded image size |
| `K`, `T_world_to_camera` | Actual calibration, never randomized/overridden |
| `distortion` | One of the explicit states below |
| `undistort_count`, `mirror_count` | Audited preprocessing count; input `mirror_count` must be 0 |
| `crop_box_xyxy` | Square or rectangular manual-RGB/detector box; use padded hand extent |
| `crop_source` | `"manual_rgb"` or `"predicted_detector"`; no evaluation-GT box |
| `crop_coordinate_system` | `"undistorted_full_image"` |
| `gt2d` | Finite 21×2 pixels in the input image coordinates |
| `gt2d_valid`, `gt2d_origin` | 21 booleans and `"independently_annotated"` or `"independent_gt3d_projection"` |
| `oracle_visibility` | Optional 21 true/false occlusion-visible values; not annotation-valid bits |
| `oracle_visibility_source` | Required when oracle visibility is supplied |
| `predicted_confidence` | Optional structure below; its absence leaves E2-5 unavailable |

Distortion forms:

```json
{"state":"raw","model":"opencv_brown5","coefficients":["k1","k2","p1","p2","k3"],"evidence":"calibration source"}
```

The quoted coefficient names above are **schema labels, not runnable numeric data**.
Replace with measured numbers. Raw inputs require `undistort_count:0`; the adapter
undistorts RGB and GT2D once, preserving the numeric K. Already undistorted inputs:

```json
{"state":"undistorted","model":"none","coefficients":[],"evidence":"explicit source declaration"}
```

Use their actual preprocessing count (0 if directly delivered as pinhole RGB, 1 if
previously corrected). No additional undistortion is applied. Unsupported fisheye
models must be converted with their documented model upstream, retaining provenance.

Predicted confidence structure: `model`, `checkpoint_sha256`,
`uses_evaluation_gt:false`, `joint_confidence` (21 numbers in [0,1]),
`visibility` (21 predicted masks/probabilities), `view_confidence` (number in [0,1]).
These must come from an actual predictor. **HaMeR does not supply a confidence
head in this checkout.** Neither an all-one placeholder nor GT validity is called
predicted confidence. Per-view confidence weights the view after per-joint
normalization, so it cannot cancel inside the joint denominator.

If GT2D is projected from dataset GT3D, geometric consistency is partly tautological.
Keep that provenance visible and independently review RGB alignment and calibration
direction. Eight paired overlays are a minimum; the adapter writes all pairs. A
GT validity flag is not an occlusion flag. Wrong-time detection uses original sync
evidence, matching IDs/times, stereo projection and visual review; matching filenames
alone cannot prove hardware synchronization.

## Commands on the existing GPU environment

```bash
cd /workspace/pose_reconstract
CUDA_VISIBLE_DEVICES=1 .venv-gpu/bin/python scripts/run_real_e2.py \
  --manifest inputs/real_e2/manifest.json --run-id real_e2_validate_NEW_ID --validate-only
```

Inspect the GT overlays and record actual visual review evidence. Then:

```bash
CUDA_VISIBLE_DEVICES=1 .venv-gpu/bin/python scripts/run_real_e2.py \
  --manifest inputs/real_e2/manifest.json --run-id real_e2_NEW_ID
```

Every directory must be new. The runner saves input/config/source snapshots, teacher
states, each final state/checkpoint/overlay, compact loss trajectories, runtime
metadata, independent metrics, and a scoped SHA256 manifest. Existing evidence is
untouched. Actual optimizer exceptions retain partial outputs and a traceback.
Neither command was run on qualified real data during the blocked session.

`configs/real_e2_v1.json` fixes E2-0…E2-8, seed 42, 800 updates, step-600 learning-rate
decay, left-teacher initialization, 3-pixel Huber scale and priors. E2-5 and E2-6 are
explicitly unavailable where their respective confidence/visibility inputs are
missing. Complete-case comparisons must use the identical sample set; per-row skips
must not be presented as the full fixed matrix. E2-S±1 needs original neighboring
right images, timestamps and annotations and is **not implemented or executed** in
this blocked delivery. No E3 gate opens automatically.

## What was actually exercised

`regress_real_e2_interface.py` reads existing `e2_synthetic_run02` arrays/checkpoint,
checks geometry/invalid-input rejection, crop reflection, normalized loss/gradients,
and independently recomputes archived metrics. It performs **zero optimizer updates**,
creates no new synthetic scene, and never claims real stereo performance.

The RGB-to-teacher-to-optimizer matrix remains **unvalidated end to end on real input**.
The standalone interface is executable; passing component regression is not
equivalent to proving the whole experimental method.
