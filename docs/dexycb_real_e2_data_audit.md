# Official DexYCB real E2 data audit

This describes the qualified input to `dex_real_e2_matrix_20261020_02`.
It establishes input consistency, not the numerical benefit of stereo optimization.

## Official sources and interpretation

Dataset: https://dex-ycb.github.io/ (CC BY-NC 4.0).
Toolkit pinned at `64551b001d360ad83bc383157a559ec248fb9100`.
Complete subject and calibration acquisition evidence:
`artifacts/dexycb_official_acquisition_20261020_01/`.

The toolkit's initializer is adapted only to enumerate the extracted subjects and
sequences. Original initializer, modified initializer and the selection-only diff
are retained in `artifacts/dex_real_e2_prepare_20261020_02/`.
Official `__getitem__` supplies RGB, depth and label filenames, color intrinsics,
and MANO side; the official NPZ labels supply pose_m, joint_2d and joint_3d.
Full NPZ labels for selected views are retained in `raw_labels/`.

Pinned official evidence:

- [Joint order and dataset interface](https://github.com/NVlabs/dex-ycb-toolkit/blob/64551b001d360ad83bc383157a559ec248fb9100/dex_ycb_toolkit/dex_ycb.py).
- [Resolution-specific color K and metadata-selected extrinsics](https://github.com/NVlabs/dex-ycb-toolkit/blob/64551b001d360ad83bc383157a559ec248fb9100/dex_ycb_toolkit/sequence_loader.py#L69-L104).
- [Millimeter depth conversion and camera-to-master point transformation](https://github.com/NVlabs/dex-ycb-toolkit/blob/64551b001d360ad83bc383157a559ec248fb9100/dex_ycb_toolkit/sequence_loader.py#L304-L326).
- [Pinhole IntrinsicsCamera construction](https://github.com/NVlabs/dex-ycb-toolkit/blob/64551b001d360ad83bc383157a559ec248fb9100/examples/render_sequence.py#L80-L88)
  and [camera-space hand-joint rendering](https://github.com/NVlabs/dex-ycb-toolkit/blob/64551b001d360ad83bc383157a559ec248fb9100/examples/render_sequence.py#L208-L229).
- [Official recording guide](https://docs.google.com/document/d/1jflhWGTn4lsGj5PiJ14Q37k0xHSnkTuDJvkdFr_j-fo),
  saved in acquisition evidence, documents RDF axes, T_wc, and approximate
  image synchronization with `slop=0.1`.

RGB is treated as **dataset-delivered pinhole projection**, with raw distortion
coefficients unknown (`null`). No undistortion is applied. This follows the delivered
K, official renderer and label projection consistency; it does not claim that raw
physical lenses have zero distortion. Official guide example coefficients are
not substituted for dataset camera measurements.

## Selected input

| Sequence | Left serial | Right serial | Baseline | Times |
| --- | --- | --- | --- | --- |
| 20200709_150654 | 836212060125 | 932122062010 | 0.637785 m | 16 |
| 20200709_150906 | 932122060861 | 932122061900 | 0.561535 m | 16 |

All 32 times are original subject-01 right-hand frames at 640×480. Each pair uses
the same official sequence and frame index. Sequence `20200709_145132` was extracted
but excluded because it uses the left hand.

The detector survey recorded 1,168 right-hand-sequence camera frames across eight
cameras. Pair selection maximized jointly detected valid-annotation frames, with
baseline as tiebreak, then sampled evenly in time. Twelve camera-frame annotations
were excluded as absent/invalid before teacher inference, with exact IDs retained.
No teacher or optimized error was used for selection. This is a detector-success
conditional benchmark, not an end-to-end detector recall benchmark.

Pretrained MediaPipe Hands 0.10.14 produces RGB landmark boxes, expanded to a square
with twice the maximum landmark extent. Model hashes and raw detections are saved.
Selfie handedness labels are swapped for original unmirrored RGB. Evaluation GT
does not define the crop. There is one physical annotated hand, reviewed visually
in both selected views.

## Coordinates and validation

Let C_L and C_R be the official camera-to-master transforms selected by sequence
metadata. The shared reference is the left camera:

`X_R = inverse(C_R) @ C_L @ X_L`.

Right GT is mapped into the shared frame with `inverse(C_L) @ C_R`. It remains
available in its original camera frame for right-camera absolute evaluation.
No extra dataset world-coordinate GT field is assumed.

GT units are meters. Mapping from official joint order to project order is explicitly
the identity permutation of 21 joints, saved in `joint_mapping.json`.
All chosen hands are right, so no input or output reflection is applied.

Validation of all 32 pairs found:

- Maximum delivered joint_3d-to-joint_2d projection error: about **0.001463 px**.
- Maximum right-GT-to-left-GT joint discrepancy: about **0.00000295 m**.
- Applying the incorrect inverse has mean error at least **0.14707 m** per pair.
- K matches decoded 640×480 images; positive camera depths and nonzero baselines pass.
- Two-view triangulation checks depth/disparity consistency in arbitrary camera axes.
  A universal horizontal disparity sign is not assumed for this nonrectified rig.
- Detector crop/full-image affine roundtrips pass; undistortion and mirror counts
  are both zero. Same sequence/time views remain in one method-validation split.

All pair checks are saved in `validation.json`. GT overlays exist for every pair.
Before optimization, contact sheets were inspected at indices
`0,5,10,15,16,21,26,31`, covering both sequences. Same-hand/action alignment,
spread fingers, bent/grasping fingers, wrist rotation, partial object occlusion
and a clipped image-edge hand were observed. Both sequences involve the same
bowl interaction; broad scene coverage and a closed-fist category are not established.

## Limits carried into the result

Matching official frame indices and visual alignment establish dataset correspondence,
not measured subframe timing. Frame/30 is nominal time only; actual per-camera
timestamps and physical offset are unavailable. ±1-frame sensitivity keeps the
original GT and shifts only the right RGB/teacher target.

Dataset joint_3d is independent of this HaMeR teacher, but cameras share the dataset
annotation process. Tiny cross-camera/projection residuals establish coordinate
consistency, not micrometer physical GT accuracy. Dataset joint_2d is consistent
with fitted 3D annotations and is not asserted to be independent manual 2D labeling.
MANO fitting bias and potential teacher pretraining overlap remain unknown.

No official per-joint oracle visibility was established. Segmentation and annotation
validity are not substituted for occlusion visibility; E2-6 is unavailable.
E2-5 uses a non-GT epipolar-distance/image-bounds heuristic, not learned visibility.
