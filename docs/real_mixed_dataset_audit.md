# Real-mixed dataset audit

Status date: 2026-10-21. Counts remain provisional until archive-level and
sample-level audits complete. A dataset cannot move to `enabled` until its
64-sample coordinate audit passes.

| Dataset | Acquisition | Images / unique times | Subjects / sequences | Hand side | 2D joints | Metric 3D | MANO global | MANO pose | Betas | K | Visibility / occlusion | Unit | License | Assignment |
|---|---|---:|---:|---|---|---|---|---|---|---|---|---|---|---|
| DexYCB | Official per-subject archives; in progress | Pending selected count | 10 subjects, 100 sequences each officially | Right hand in current protocol; verify every sequence | Yes | Yes | Yes (`pose_m`) | Yes (`pose_m`) | Yes (subject calibration) | Yes | 2D validity and object segmentation available | meters | CC BY-NC 4.0 | subjects 02-06 train, 07 dev, 08-09 sealed Final-E4, 01 reused diagnostic |
| FreiHAND v2 | Official 3,881,833,583-byte archive; ZIP CRC and SHA-256 verified | 130,240 official images / 32,560 unique captures; one RGB version plus masks stored remotely | 32 subjects reported; subject identity is not released per image | Right | Derived from K and 3D; no independent visibility flag | Yes | Axis-angle, audited | 45 axis-angle values, audited | 10 values, audited | Yes | Segmentation mask; no independent per-joint occlusion flag | meters | Research only, commercial use prohibited | train only; one capture per index with online background replacement |
| RHD v1.1 | Complete and previously audited | 41,258 usable train images plus isolated dev | Synthetic scenes | Left and right | Yes | Yes | No reliable MANO parameters | No reliable MANO parameters | No | Yes | Per-joint visibility | meters after verified conversion | Research only | minority train auxiliary, at most 13.33% |
| HO3D v3 | Not acquired | 66,034 reported train images | 55 reported train sequences | Predominantly right | Yes | Yes | Yes | Yes | Verify | Yes | Hand-object occlusion | Verify | Official terms pending local acceptance | unavailable in first round: archive about 66.8 GB |
| H2O3D | Not acquired | Pending | Pending | Both, verify | Verify | Verify | Verify | Verify | Verify | Verify | Hand-object occlusion | Verify | Pending | unavailable in first round |
| InterHand2.6M | Not acquired | 2,590,347 total; 1,410,699 single-hand reported | Multiple captures and subjects | Left and right | Yes | Yes | Fitted MANO available | Fitted MANO available | Fitted MANO available | Yes | Joint validity; interacting-hand occlusion | millimeters in official annotations, verify | Official terms pending | unavailable in first round: images about 80 GB |

## Required coordinate audit

For every enabled dataset, sample at least 64 rows before training and save:

1. original-image GT overlay;
2. crop-space GT overlay;
3. metric 3D projected through the declared K;
4. crop/full-image affine round trip;
5. canonical left-hand mirror and joint order;
6. MANO-generated versus dataset joints when MANO parameters are valid;
7. wrist/root identity and fingertip convention;
8. metric-unit checks and global-orientation frame checks.

The report must contain mean, median, P95 and maximum projection error, valid
joint counts, and links to all rejected examples. Reporting only the maximum is
not sufficient.

FreiHAND's 64-capture audit is stored in
`artifacts/freihand_v2_audit_20261021_01`. Its original JSON incorrectly named
the final three MANO-fit values as xyz translation. The verified layout is
3 global axis-angle + 45 articulated axis-angle + 10 betas + 2 `uv_root` +
1 scale. Training excludes the final three values. The audit script now emits
the corrected names. Independent MANO reconstruction found a 2.96 mm mean,
8.44 mm P95 and 14.18 mm maximum joint discrepancy, so metric joints and MANO
parameters are supervised independently rather than treated as identical
joint targets.

## Label-mask policy

Every training row carries independent booleans for:

- `valid_2d`;
- `valid_3d`;
- `valid_global_orient`;
- `valid_hand_pose`;
- `valid_betas`;
- `valid_camera`;
- `valid_visibility`.

A zero numeric tensor is allowed only as storage for a masked-out field. It must
never be interpreted as a valid target. Effective sample and scalar counts are
logged for every loss component.

## Sampling policy

The first-round available-domain mix is DexYCB 53.33%, FreiHAND 33.33%, and RHD
13.33%. M-S0 realizes this exactly as 16/10/4 examples in every 30-example
batch. The remote FreiHAND copy stores one image per unique capture and applies
mask-based online background replacement instead of storing four background
variants. Later stages additionally bucket by DexYCB subject/sequence,
occlusion and pose diversity so raw dataset size cannot dominate sampling.
