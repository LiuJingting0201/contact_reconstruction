# Real E2 data investigation (in progress, 2026-10-20)

## Inherited evidence

Complete latest experiment record (376 lines), complete reproduction protocol (990 lines), current E1/synthetic-E2 runner/verifier, effective configs, full per-sample result JSONs and `artifacts/verification.json` were read before new experiment work. `scripts/audit_real_e2_start.py` ran once, independently matching all 446 inherited evidence hashes. Its new output directory is `artifacts/real_e2_audit_20261020_01`; it stores before snapshots of scripts/docs/configs, artifact inventory (1,258 pre-existing files), NPZ field/shape/finite inventories, current-vs-run script diffs, and unavailable git status.

Both local project and `/workspace/pose_reconstract` are not git repositories. Do not invent a commit or claim a clean git diff. Prior archives have pinned source revisions and source hashes. No prior optimization/training was rerun.

Remote disk: 24 GB available on project filesystem; `/mnt/session/{work,outputs,docker-probe}` contains no input files in the inspected depth. No dedicated `/data` or `/datasets` mount exists. Searched dataset-like directories in `/workspace`, `/mnt/session`, `/opt/simready`, `/ebs/docker` to depth 4; only project model assets, packaged evaluation NPZ files, unrelated IsaacLab assets and software directories found. Packaged Fast-HaMeR evaluation NPZ files exist but no paired real image dataset has been established.

## Candidate evidence and retrieval

### DexYCB (preferred current candidate)

- Official: https://dex-ycb.github.io/ ; license CC BY-NC 4.0. This task is research method verification; retain attribution and original terms.
- Paper: https://dex-ycb.github.io/assets/chao_cvpr2021.pdf : eight temporally synchronized, extrinsically calibrated RealSense D415 RGB-D cameras; 640×480, 30 fps. Human per-view 2D annotations and RGB-D optimization produce independent dataset hand pose annotations. They are not HaMeR pseudo GT. Annotation noise and fitted-MANO bias remain limitations.
- Toolkit README read: https://raw.githubusercontent.com/NVlabs/dex-ycb-toolkit/master/README.md . Labels contain `joint_3d` [1,21,3] in each camera coordinate system, `joint_2d` [1,21,2], `pose_m` [1,51] (PCA MANO plus translation), `seg`, `pose_y`. Invalid joints can be all -1. Need further verify units, distortion, extrinsics direction, metadata and visibility semantics from source.
- Official calibration download **succeeded**, 14,464 bytes, at `artifacts/real_e2_audit_20261020_01/data_search/dex_calibration.raw` (probably tar.gz; inspect). URL: https://drive.google.com/uc?export=download&id=1UAwVKT4Rgb1fLcFoa1o71_-0NtSvvLAQ .
- Official image packages are 12 GB/subject tar.gz; do not download full package.
- Public mirror `UCBProject/DexYCB`, revision `95442e7b43f8282e41cd8b97aea2b5eb848e5b35`, has individual RGB files under `20200918-subject-06`. API https://huggingface.co/api/datasets/UCBProject/DexYCB/tree/main/20200918-subject-06 .
- Available sequences: `20200918_113738`, `114502`, `114814`, `114840`, `115412`, `115634`, `115755`, `115826`, `120449`, `120600`, `121129`. First sequence contains eight camera directories `836212060125`, `839512060362`, `840412060917`, `841412060263`, `932122060857`, `932122060861`, `932122061900`, `932122062010`; sequence root API showed no meta.yml, so annotations likely need a second archive/source.
- First image exists: `20200918-subject-06/20200918_113738/836212060125/color_000000.jpg`, size 111080, SHA256 `96f897842a7712b651d6156b2caacf4ce640bec19a2b6d9847d34c95082bc2e0`. Still need actual binary download/inspect and more than one view.
- Mirror `Louis0411/dexycb_mesh`, revision `18376c6f7ab2f1cda6f5d316326a67f886171514`, CC BY-NC 4.0, strips images but retains labels/calibration. `tars/20200918-subject-06.tar.gz` size 127638298, SHA256 `7e7740426cc009dd4b389d908c64a06d2805d2be08f9549070e1739aa1930774`. Could stream extract only relevant metadata/labels and stop once acquired. `models_calibration.tar.gz` is 1.636 GB and unnecessary because official 14 KB calibration is already downloaded.
- https://huggingface.co/api/datasets/Louis0411/dexycb_mesh/tree/main/tars gives all archive metadata.
- Need inspect mirror README and label integrity against official conventions; mirror presence alone does not establish usable input.

### InterHand2.6M (fallback)

- Official complete page read: https://mks0601.github.io/InterHand2.6M/ .
- Calibrated multi-view real capture, image metadata has capture/subject/sequence/camera/frame index, world XYZ in millimeters, camera positions/rotations/focal/principal point. Joint-valid masks do NOT necessarily mean occlusion.
- Prefer human-annotated subset for independence; MANO NeuralAnnot fits are not independent of those dataset labels.
- Official repo LICENSE is CC BY-NC 4.0: https://raw.githubusercontent.com/facebookresearch/InterHand2.6M/main/LICENSE .
- Images are uncompressed split tar chunks (1.9 GB each, ~80 GB total). Index read: https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/InterHand2.6M/InterHand2.6M.images.5.fps.v1.0/index.html . A range-based extractor may be possible but needs an index.
- Google annotation folder HTML successfully saved at `data_search/interhand_annotations.raw`, URL https://drive.google.com/drive/folders/12RNG9slv9i_TsXSoZ6pQAq-Fa98eGLoy ; full GitHub root API saved as `interhand_github.raw`.
- Distortion/undistortion statement remains unresolved.

### STB (lower priority)

- Official README saved at `data_search/stb_readme.raw`; correct repo https://github.com/zhjwustc/icip17_stereo_hand_pose_dataset .
- Real Bumblebee2 stereo plus F200 depth simultaneously; 120.054 mm baseline, fx=fy=822.79041, cx=318.47345, cy=250.31296. 18,000 frames in 12 sequences.
- GT first joint is **palm center, not wrist**. This conflicts with wrist error without a documented mapping; do not invent wrist GT.
- Source README offers OneDrive/Baidu links but did not establish license or explicit undistortion. Thus not yet qualified.

## Network/tool observations

Remote urllib to Hugging Face failed with `[Errno 101] Network is unreachable`. Local system-Python urllib calls can take minutes while TRAE sandbox prints Crashpad errors, even with short socket timeout; eventually official calibration/README/folder pages were downloaded. Check actual saved files before concluding failure or retrying. Local long sessions 7204/7205 were stopped; curl session 7209 handle expired without retained output. WebFetch can read HTTP text/API metadata but cannot download binary artifacts. Prefer local downloaded files plus rsync when remote access fails.

## Implementation correction identified

Old `run_mano_backend.py` normalizes per view but sums across views. New real-E2 must average across active views (or normalized explicit view weights), handling missing observations safely. Preserve old source/results; create separate real-E2 modules and unique runs. Old code also lacks per-sample timing, full metrics distributions, held-out-view projection for single-view fits, and reload evidence for optimized states. All are required for real E2.

## Retrieval outcome and corrections

The official calibration archive is valid gzip tar. It contains eight 640×480
intrinsics, per-day extrinsics and subject MANO calibrations. Color K has no
distortion fields. Explicit undistortion provenance remains unresolved.

Official `SequenceLoader` uses the supplied extrinsics to map camera point clouds
to world, and its inverse for world-to-camera:
https://raw.githubusercontent.com/NVlabs/dex-ycb-toolkit/master/dex_ycb_toolkit/sequence_loader.py
Maintainer confirmation:
https://github.com/NVlabs/dex-ycb-toolkit/issues/6#issuecomment-855322639 .
Thus **invert DexYCB extrinsics once** for the new canonical interface.

The first UCBProject JPEG downloaded and decoded successfully (640×480; hash as
above), but has no usable visible hand. It is only a transport probe, never a
selected evaluation frame. The annotation-only mirror now returns HTTP 401 with
`Invalid username or password.` on both API and binary range requests; body and
headers are retained as `labels_probe.bin` and `labels_probe_headers.txt`.
Rechecking did not resolve this.

A 65,536-byte range of UCBProject's uncompressed 5,094,860,800-byte subject06 tar was
retrieved, not the full dataset. `tar_probe.bin` contains the root, sequence
`20200918_114647`, camera `839512060362`, and the start of `color_000011.jpg`.
This proves byte-range retrieval works, but does not establish label availability.
No paired GT subset was obtained. Do not characterize all public data as inaccessible.

Final disposition: input qualification blocked; **no real E2 executed**.
New data contract: `docs/real_e2_input_contract.md`. Component regression reads
existing synthetic arrays/checkpoint only, performs zero optimizer updates, and is
not a new synthetic experiment or a real stereo result.
