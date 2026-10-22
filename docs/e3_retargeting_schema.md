# Cached ViT+B2 retargeting schema, version 1.0

The export contains all 32 cached synchronous DexYCB samples. No inference,
optimization, GT selection, filtering, interpolation or retargeting is performed.
The baseline remains HaMeR ViT plus frozen B2.

`retargeting.json` contains `schema_version`, `pipeline`, `joint_names`, and
`records`. `retargeting.npz` is readable with `numpy.load(..., allow_pickle=False)`.
The sample order is identical across JSON, NPZ and the original Real-E2 cache.

| Field | NPZ shape | Meaning |
|---|---|---|
| sample_id, sequence_id | (32,) | Stable source identity; sequence separates timestamp domains |
| frame_index, timestamp_s | (32,) | Frame index and nominal sequence-relative seconds, frame/30 |
| joints_left_camera_m | (32,21,3) | Right-hand MANO joints in left RGB-camera coordinates, meters |
| joints_wrist_local_m | (32,21,3) | Previous field minus wrist joint; axes remain left-camera axes |
| global_orientation_matrix | (32,3,3) | MANO canonical-to-left-camera global rotation, column-vector convention |
| articulated_rotation_matrices | (32,15,3,3) | Frozen MANO internal rotations in MANO order |
| shape_betas | (32,10) | Frozen MANO shape |
| mano_translation_left_camera_m | (32,3) | MANO model translation; use joint 0 for anatomical wrist position |
| K | (32,2,3,3) | Full-image intrinsics; view order left, right |
| T_left_to_camera | (32,2,4,4) | `p_camera = R p_left + t`; translation in meters |
| camera_serial | (32,2) | Corresponding left/right camera identity |
| confidence_available | (32,) | False; no calibrated or learned confidence exists |
| quality_flag | (32,) | Non-GT finite/positive-depth validity, not an accuracy guarantee |
| evaluation_only_quality_failure_gt_assisted | (32,) | Prior benchmark audit label; never a deployment confidence or filtering rule |

JSON additionally stores handedness, synchronization identity, reference-frame
name, timestamp kind, `confidence: null`, calibration distortion metadata, source
image/extrinsics SHA256 and full-image detector crops. It stores the same arrays
without rounding. Float32 joints and rotations are preserved exactly through
JSON/NPZ round-trip; K/T preserve the float64 evaluation-cache values. Cached
optimization states also contain their original float32 solver K/T.

Joint order is wrist, followed by four joints of thumb, index, middle, ring,
little: MCP/PIP/DIP/tip. These names are indexing labels; thumb anatomy uses
CMC/MCP/IP rather than the generic labels. Joint 0 is wrist; fingertips are
4/8/12/16/20. MANO rotation order is its own native order, not this 21-joint order.

The camera uses x right, y down, z forward. Wrist-local means translation removal
only, without rotation alignment or scale normalization. Robot conversion must
declare the robot coordinate frame and joint mapping separately.
Timestamps are nominal, not measured sensor capture times or wall-clock UTC.
Same-frame synchronization comes from the dataset export.

All 32 records remain in the output, including the one GT-assisted quality
failure. HaMeR has no learned confidence output; numerical/depth validity must
not be interpreted as predicted accuracy.

`cached_evidence/sample_*/` preserves both-view teacher arrays, B0/B2 states and
compact runtime/loss trajectories, sample metadata and evaluation targets.
GT inside this evidence is for audit only. `baseline_metrics.json` preserves
sample-level and mean/median/P95 metrics. `source_hashes.json`, `sha256.json`
and `verification.json` document equality and integrity.
