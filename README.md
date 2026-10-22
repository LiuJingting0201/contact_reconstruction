# Contact Reconstruction Experiments

This repository contains the experiment code, configurations, and public
process records for 3D hand-pose reconstruction studies based on Fast-HaMeR.

## M-S1 Real Mixed No-KD

The completed M-S1 study trains a ResNet-50 model from ImageNet1K V2
initialization on a real mixed dataset:

- DexYCB as the primary real domain
- FreiHAND for MANO pose and shape supervision
- RHD as an auxiliary domain
- no teacher, knowledge distillation, or adversarial loss

DexYCB is split by subject and sequence. Subject 02 is used for training,
subject 07 for development, and subjects 08/09 remain sealed for Final-E4.
The registered sampler correction changes only the per-batch domain
composition from `16/10/4` to `12/10/8` for DexYCB/FreiHAND/RHD.

The correction run completed 10,000 updates. On the independent DexYCB
development set, its selected checkpoint improved the same-dev RHD-only
baseline as follows:

| Metric | RHD-only baseline | M-S1 correction |
|---|---:|---:|
| Wrist-relative MPJPE | 64.326 mm | 30.661 mm |
| Fingertip error | 84.551 mm | 42.100 mm |
| Visible 2D EPE | 29.197 px | 19.653 px |

All development domains had zero collapse, negative-depth, and nonfinite
failure rates at the selected checkpoint. M-S2 was not started because the
strict preregistered gates were not all met, including a remaining RHD 2D
regression.

See
[`artifacts/m_s1_final_comparison_20261022/comparison.md`](artifacts/m_s1_final_comparison_20261022/comparison.md)
for the final two-run comparison.

## Layout

- `scripts/`: data audits, preparation, training, reporting, and verification
- `configs/`: preregistered experiment configurations
- `experiments/`: supporting experiment and dependency specifications
- `docs/`: experiment plans, protocols, audits, and chronological records
- `artifacts/`: curated aggregate reports only

Datasets, checkpoints, per-sample predictions, frozen split rows, raw logs,
third-party source trees, and local environments are intentionally excluded.

## Reproduction Notes

The scripts expect the dataset and Fast-HaMeR paths recorded in the JSON
configuration files. Remote runner scripts use the original
`/workspace/pose_reconstract` layout and should be adapted for another host.
The saved verification scripts recompute metrics from local prediction
artifacts, but those large prediction files are not stored in Git.
