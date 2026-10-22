# Real-mixed no-KD data protocol

## M-S1 scope amendment (2026-10-21)

The latest user instruction supersedes the earlier prospective stages below.
For this M-S1 run, train uses **subject-02 only**, development uses **subject-07**,
and the previously sealed Final-E4 subjects remain **08/09**. All usable frames
and all eight RGB cameras of the two available full subject archives are
eligible; the M-S0 fixed subset is retired for formal training. No sequence is
randomly split across roles.

The executable registration is
`configs/resnet50_real_mixed_m_s1_subject02.json`. Peak LR is 3e-5, linear warm-up
is 500 updates, cosine decay ends at 0.1 times peak LR, AdamW uses
betas=(0.9,0.999), eps=1e-8 and weight decay=1e-4, and gradient norm clipping
is 1.0. Seed is 20261021. The budget is 10,000 updates, with complete dev
evaluation every 1,000 updates and a patience of five evaluations. Positive
camera scale remains softplus(raw)+0.05. The model starts from ImageNet1K V2
plus a new head, not an M-S0 or RHD checkpoint.

The selection criterion is minimum full subject-07 dev wrist-relative MPJPE,
with at least 0.1 mm improvement and zero failure across all dev domains.
Two consecutive dev evaluations with nonzero collapse in any one domain
stop training; initial step zero participates. Nonfinite predictions also stop
training. All early failures remain in the logs. M-S0 FreiHAND collapse at
step 100 (9.375%) remains historical plumbing evidence, not an accuracy result.

FreiHAND uses a deterministic SHA256 split of original capture IDs (10% dev),
with all four background variants of each capture assigned together. Published
subject/sequence IDs are unavailable: this is capture-disjoint, not verified
subject- or sequence-disjoint. DexYCB subject-07 is the independently held-out
real-domain test for this stage's development decisions. RHD retains its
existing nonoverlapping training/evaluation-image dev split. All available
training samples are eligible; a fixed update budget does not guarantee a full
epoch in every auxiliary domain.

Before model access, the pipeline saves exact train/dev/final inventories,
per-image/annotation SHA256, subject/sequence/frame fields, and config hashes,
then audits 64 samples per domain and split. Missing identity metadata remains
null. DexYCB PCA articulated pose and subject betas remain masked.
Annotation-availability buckets are only proxies; they do not establish actual
occlusion ground truth.

The RHD-only formal model is evaluated on the same full dev set for a paired
comparison. A 2,000-replicate sequence-cluster bootstrap reports 95% confidence
intervals for wrist and fingertip improvements. The preregistered substantial
improvement margin is 10% below historical thresholds 32.119 mm and 43.076 mm;
2D must improve by 10% below 15.712 px or decrease across the last three dev
evaluations. A greater-than-25% regression in an auxiliary domain is reported
as imbalance and disqualifies automatic M-S2 eligibility.

Stop after M-S1 and freeze best/last checkpoints. **Do not execute M-S2, old E3,
Final-E4 inference, KD, B2 tuning, Dyn-HaMR integration or retargeting.** Earlier
prospective language allowing E3/E4 after freeze does not authorize them in
this stage.

## Earlier parent registration

Protocol frozen on 2026-10-21 before any mixed-data training or Final-E4
inference.

## Split

| Role | DexYCB subjects | Model access |
|---|---|---|
| Train | 02, 03, 04, 05, 06 | Training only |
| Development | 07 | Checkpoint selection |
| Final-E4 | 08, 09 | One run after all decisions freeze |
| Reused E3 diagnostic | 01 | One historical comparison after freeze |

This follows the official DexYCB `s1` unseen-subject split except that subject-01
is removed from training because its 32-pair E3 subset has already been observed.
Subject-10 is not used in Final-E4 because the official `s1` protocol assigns it
to training.

Final-E4 contains 500 synchronized stereo times: 250 from subject-08 and 250
from subject-09. Each subject contributes 25 uniformly spaced official
sequences and 10 uniformly spaced GT-valid times per sequence. Candidate
selection uses only sequence metadata and frame indices. Validation may reject
nonfinite labels, nonpositive depth, fewer than four valid 2D joints, pinhole
reprojection error at or above 2 px, or stereo 3D disagreement at or above
2 mm. No RGB content, prediction, checkpoint, model error, or B2 result
participates in selection.

The fixed stereo cameras are `836212060125` and `932122062010`. Their baseline
is 0.634 m under the subject-08 official calibration. These are also the cameras
used by the reused E3 diagnostic.

Each sealed subject manifest records:

- official URL, license, archive byte count, SHA-256 and full gzip CRC pass;
- exact selection rule and all candidate GT validation outcomes;
- sample, subject, sequence, frame, camera and object/grasp identifiers;
- SHA-256 and byte count of every extracted RGB, label, metadata and calibration
  file;
- manifest SHA-256 and read-only filesystem permissions.

Training and development code must reject paths under the Final-E4 root.

## Availability

| Dataset | Current decision | Reason |
|---|---|---|
| DexYCB | Enable | Official per-subject archives; CC BY-NC 4.0; complete camera, metric 3D and MANO labels |
| FreiHAND | Enable | Official 3,881,833,583-byte training archive; 130,240 RGB samples; K, metric 3D and shape labels |
| RHD | Enable as minority auxiliary | Existing complete audited training set; synthetic data capped at 13.33% in the first available mix |
| HO3D/H2O3D | Disable in first round | Complete HO3D v3 is about 66.8 GB, beyond the 26 GB remote free-space budget |
| InterHand2.6M | Disable in first round | Official image files require about 80 GB |

Unavailable datasets are not represented by zero labels or synthetic
placeholders. The available target weights are the preregistered
DexYCB/FreiHAND/RHD weights renormalized from 40/25/10 to
53.33/33.33/13.33 percent.

FreiHAND is research-only and prohibits commercial use. DexYCB is CC BY-NC 4.0.
RHD is retained under its original research-only terms. Dataset citations and
source URLs must accompany exported checkpoints.

## Supervision

Every label family has an explicit validity mask. Missing MANO global
orientation, articulated pose or betas never becomes a valid zero target.
Losses are normalized by their own valid scalar counts:

- visible 2D joint reprojection;
- wrist-relative metric 3D joints;
- fingertip-weighted wrist-relative 3D;
- bone-direction angular loss;
- MANO global orientation geodesic loss;
- articulated MANO pose geodesic loss;
- betas loss;
- weak-camera reprojection consistency.

Absolute root depth is not a primary training target because frozen B2 estimates
translation from stereo pseudo-2D and real calibration.

## Execution gates

M-S0 runs 500 updates and must show finite component losses, nonzero backbone
and head gradients, valid masks for every enabled domain, positive camera scale,
noncollapsed projections, strict checkpoint reload and per-domain overlays.

M-S1 runs a fixed 10,000-update budget with evaluation every 1,000 updates and
patience of five evaluations. M-S2 is allowed only if M-S1 improves a majority
of wrist-relative MPJPE, fingertip error, real-domain 2D EPE and failure rate
without projection collapse.

Only after checkpoint, configuration and all decisions freeze may the workflow
run the reused E3 diagnostic once and then Final-E4 once. B2 remains unchanged.
