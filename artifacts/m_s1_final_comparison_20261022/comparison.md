# M-S1 final comparison

Both runs completed and passed saved-artifact verification. The second run is the single registered sampler correction. Each checkpoint was selected using its own full independent DexYCB dev evaluations and the same safety rule.

## Independent dev results

| Domain | Model | Wrist MPJPE mm | Fingertip mm | Visible 2D px | Failure |
|---|---|---:|---:|---:|---:|
| dexycb | RHD-only baseline | 64.326 | 84.551 | 29.197 | 0.0000% |
| dexycb | source | 31.477 | 43.151 | 20.170 | 0.0000% |
| dexycb | correction | 30.661 | 42.100 | 19.653 | 0.0000% |
| freihand | RHD-only baseline | 55.339 | 76.477 | 19.475 | 0.0000% |
| freihand | source | 26.829 | 42.664 | 14.597 | 0.0000% |
| freihand | correction | 26.186 | 41.993 | 14.184 | 0.0000% |
| rhd | RHD-only baseline | 32.119 | 43.076 | 15.713 | 0.0000% |
| rhd | source | 42.020 | 54.484 | 24.205 | 0.0000% |
| rhd | correction | 38.500 | 50.203 | 22.754 | 0.0000% |

The historical 32.119 mm / 43.076 mm / 15.712 px values reproduce on RHD dev, not on the frozen DexYCB subject07 dev. They remain unchanged as user-requested absolute gates. Improvement estimates use the RHD-only checkpoint evaluated on the same full DexYCB dev.

## Per-run conclusions

### source

Run: `resnet50_real_mixed_m_s1_subject02_20261021_01`. Full metrics and curves: [results.md](../resnet50_real_mixed_m_s1_subject02_20261021_01/results.md).
Stop: `budget_completed` at 10000 updates; best-dev: 9000. M-S2 eligible: **False**.
All three DexYCB dev errors are lower than the same-dev baseline: **True**.
DexYCB improvement over the same-dev RHD-only baseline: wrist 51.07%, fingertip 48.96%, 2D 30.92%.
Failed gates: wrist_below_historical_by_10pct, tips_below_historical_by_10pct, no_severe_domain_imbalance.
Severe regression: rhd/wrist_relative_mpjpe_mm, rhd/fingertip_wrist_relative_error_mm, rhd/visible_2d_epe_px.
Training collapse history: `{"affected_updates": 18, "first_update": 281, "last_update": 581, "maximum_batch_rate": 0.06666667014360428, "raw_evidence": "steps.jsonl; retained independently of later dev recovery"}`.
Descriptive paired 2D change: `{"mean_paired_delta_px": -9.027045554310591, "sequence_cluster_bootstrap_95pct_ci": [-9.848227938931338, -8.212893744248833], "clusters": 100, "replicates": 2000, "interpretation": "Descriptive, post-selection; one dev subject, no subject-level CI."}`.

- last.pt SHA256: `e1ae9fd6458290b26b95a9a56e5e40eae15e84e0b39d7ec94eee437692c8ad03`
- best_dev.pt SHA256: `1d0457e066cbd2057d7492eafb25a17dc4b0d086f4a0fdb79346baedc3331a89`

### correction

Run: `resnet50_real_mixed_m_s1_sampler_correction_20261021_01`. Full metrics and curves: [results.md](../resnet50_real_mixed_m_s1_sampler_correction_20261021_01/results.md).
Stop: `budget_completed` at 10000 updates; best-dev: 10000. M-S2 eligible: **False**.
All three DexYCB dev errors are lower than the same-dev baseline: **True**.
DexYCB improvement over the same-dev RHD-only baseline: wrist 52.34%, fingertip 50.21%, 2D 32.69%.
Failed gates: wrist_below_historical_by_10pct, tips_below_historical_by_10pct, real_dev_last_three_evaluations_improving, no_severe_domain_imbalance.
Severe regression: rhd/visible_2d_epe_px.
Training collapse history: `{"affected_updates": 14, "first_update": 302, "last_update": 767, "maximum_batch_rate": 0.03333333507180214, "raw_evidence": "steps.jsonl; retained independently of later dev recovery"}`.
Descriptive paired 2D change: `{"mean_paired_delta_px": -9.544904880546964, "sequence_cluster_bootstrap_95pct_ci": [-10.286587084204058, -8.813751869751295], "clusters": 100, "replicates": 2000, "interpretation": "Descriptive, post-selection; one dev subject, no subject-level CI."}`.

- last.pt SHA256: `4416633ceec579725fa39678a64fbd95b2ad0f034e1d52e97ad857cbc353ca09`
- best_dev.pt SHA256: `74e2e2a287cc4fd4e83c7f98414ce46a618fbb474e56dff4bcd285f0e4296795`

## Train/dev diagnostic gap at selected checkpoints

| Run | DexYCB train diagnostic wrist mm | Full DexYCB dev wrist mm | Gap mm |
|---|---:|---:|---:|
| source | 10.350 | 31.477 | 21.127 |
| correction | 11.121 | 30.661 | 19.540 |

The training diagnostic uses 256 fixed, unaugmented samples per domain and never selects checkpoints. It is a diagnostic panel, not a full-training-set estimate. Independent dev improvement does not eliminate the remaining train/dev gap or establish generalization beyond this dev subject.

## Interpretation and limits

Geometry confidence intervals are reported in each run's results.json. The additional 2D intervals above are descriptive and were computed after checkpoint selection. All intervals cluster by sequence within one development subject; they do not establish cross-subject generalization or adjust for checkpoint selection.
The full training pools were eligible; finite update budgets do not guarantee every sample was drawn. Actual exposure is reported per domain in each run.
The correction changes batch composition from DexYCB/FreiHAND/RHD 16/10/4 to 12/10/8. It uses fresh initialization with unchanged losses, optimizer and schedule. An NPZ metadata caching optimization was applied before the correction; the original training script snapshot is retained.
No third correction or M-S2 was started. Final-E4 remains sealed. No E3, KD, B2 tuning, Dyn-HaMR integration or retargeting was run.
