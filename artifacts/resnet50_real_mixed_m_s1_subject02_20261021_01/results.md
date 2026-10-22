# M-S1 real mixed no-KD result

M-S2 eligible: **False**. No M-S2 execution was performed.
Stop reason: `budget_completed`. Last update: 10000; selected update: 9000.

## Independent dev results

| Domain | Model | Wrist MPJPE mm | Fingertip mm | Visible 2D px | Failure |
|---|---|---:|---:|---:|---:|
| dexycb | RHD-only | 64.326 | 84.551 | 29.197 | 0.0000% |
| dexycb | M-S1 best | 31.477 | 43.151 | 20.170 | 0.0000% |
| freihand | RHD-only | 55.339 | 76.477 | 19.475 | 0.0000% |
| freihand | M-S1 best | 26.829 | 42.664 | 14.597 | 0.0000% |
| rhd | RHD-only | 32.119 | 43.076 | 15.713 | 0.0000% |
| rhd | M-S1 best | 42.020 | 54.484 | 24.205 | 0.0000% |

The historical 32.119 mm / 43.076 mm / 15.712 px values reproduce on RHD dev, not on the frozen DexYCB subject07 dev. They remain unchanged as user-requested absolute gates. Improvement estimates use the RHD-only checkpoint evaluated on the same full DexYCB dev.

## Gate checks

- wrist_below_historical_by_10pct: False
- tips_below_historical_by_10pct: False
- wrist_same_dev_paired_ci_below_zero: True
- tips_same_dev_paired_ci_below_zero: True
- 2d_improved_or_stably_decreasing: True
- all_domain_failure_at_most_0_1pct: True
- no_collapse_stop: True
- no_nonfinite_stop: True
- dev_geometry_improved_from_initial: True
- real_dev_last_three_evaluations_improving: True
- no_severe_domain_imbalance: False

## Projection-collapse history

M-S0 step 100 FreiHAND collapse was 9.375%; that early failure remains in the record. M-S0 metrics are not accuracy evidence. All M-S1 dev failures, including recoveries, are retained in `evaluations.jsonl` and `results.json`.
Training batches with collapse: 18; maximum batch rate: 6.6667%. Raw per-update evidence remains in `steps.jsonl`.

## Training pool coverage

| Domain | Eligible | Unique drawn | Sample draws | Effective passes |
|---|---:|---:|---:|---:|
| dexycb | 55762 | 55762 | 160000 | 2.869 |
| freihand | 117076 | 100000 | 100000 | 0.854 |
| rhd | 61895 | 40000 | 40000 | 0.646 |

All eligible samples were available to the sampler. Draw counts include attempted batches if training stopped before an optimizer update.

## Scope and limitations

DexYCB uses subject02 for train, subject07 for full independent dev, and sealed subjects08/09 for Final-E4. No E4 image/label/model access, E3, KD, B2 tuning, Dyn-HaMR integration or retargeting occurred. FreiHAND is capture-disjoint with all four RGB variants grouped together; public subject/sequence IDs are absent, so subject independence cannot be claimed. Train diagnostic panels select no checkpoints. RHD-only baseline was evaluated using exactly the same full dev crops and masks. DexYCB dev contains one independent subject; sequence bootstrap intervals do not measure uncertainty across subjects. Rotation alignment is wrist-anchored proper rotation, without scale fitting. Per-finger errors exclude the common wrist. The aggregate is an unweighted domain mean.

Severe domain regression requires sampler diagnosis and a separately registered follow-up; a favorable aggregate cannot override it.

## Checkpoint SHA256

- `last.pt`: `e1ae9fd6458290b26b95a9a56e5e40eae15e84e0b39d7ec94eee437692c8ad03`
- `best_dev.pt`: `1d0457e066cbd2057d7492eafb25a17dc4b0d086f4a0fdb79346baedc3331a89`

See `splits/`, the copied JSON configuration, `curves.csv`, and PNG curves.
