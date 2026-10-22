# Debug Session: hand-camera-collapse

- **Status**: [VERIFIED — instrumentation retained; reconstruction accuracy remains insufficient]
- **Issue**: RHD ResNet/MANO pilot projection collapse and inaccurate instance-specific hand orientation.
- **Debug Server**: `http://127.0.0.1:7777/event`; remote GPU reports through SSH reverse tunnel `127.0.0.1:17777`.
- **Log File**: `.dbg/trae-debug-log-hand-camera-collapse.ndjson`
- **Authorization**: User explicitly requests debugging, corrections and continued experiments against the meeting transcript. Reproduction and post-fix runs are automated; do not pause for manual reproduction. Retain instrumentation/evidence; do not clean up without confirmation.

## Reproduction Steps

1. Preserve v2 run02/v3 artifacts and source snapshots.
2. Instrument actual model initialization, camera outputs, per-loss gradients and coordinate checks.
3. Reproduce v2/v3 on physical GPU 1 under the existing environment.
4. Use runtime evidence to select the smallest corrective change.
5. Run controlled post-fix training and teacher/geometric checks matching the meeting claims.

## Hypotheses & Verification

| ID | Hypothesis | Evidence required | Status |
|---|---|---|---|
| A | Camera head starts at an inappropriate scale/translation and reprojection optimization can shrink its projected extent | Actual initial/final scale, camera translation, depth, per-head gradients; positive-scale fitting control | Confirmed invalid scale/depth; initialization is a contributing mechanism |
| B | A coordinate, handedness, joint-map or MANO rotation-convention mismatch prevents orientation learning | GT crop projection checks, exact MANO self-reconstruction, per-instance pose-fit control | Gross chain failure not supported: synthetic <0.1 mm; eight RHD GT fits mean 6.024 mm; anatomical/local-minimum residuals remain |
| C | Current 3D/2D objective balance gives conflicting or weak gradients for orientation/camera | Per-loss gradient norms and cosine similarity by head; matched loss comparison | Imbalance confirmed; 10x reweighting has no consistent 2D gain in deterministic control |
| D | Small data coverage and limited optimization, rather than an implementation defect, explain residual accuracy | Single/few-image GT overfit and matched train/development behavior | 1000 steps improve train 2D to 14.286 px but worsen development to 40.405 px; supports overfitting |
| E | The pilot deviates from the meeting's central requirement of accurate 2D plus geometric postprocessing | Pretrained teacher 2D accuracy, real-K initialization and pseudo-label fitting measurements; no unsupported stereo claim | Teacher 7.185 px; real-K pseudo fitting improves 2D but worsens mean depth; full MANO/stereo remains untested |

## Meeting anchors

- `docs/reference/meeting_transcript.md:12–17`: nominal-5000 2D pseudo-observation, real-focal initialization, reprojection and stereo losses. Accuracy of either branch is a claim to measure.
- `docs/reference/meeting_transcript.md:32`: train ResNet without KD; detector retraining is separate.
- `docs/reference/meeting_transcript.md:42`: small model needs accurate 2D for downstream optimization, not necessarily best standalone 3D.

## Log Evidence

The first instrumentation launch had a syntax error (one missing dictionary brace), before model construction or artifacts. Failed launcher and command logs are retained; corrected instrumentation passed Python compilation.

38 runtime events collected in the unchanged two-run reproduction:

- Line 1: actual first optimizer parameter is `mano_head.transformer.pos_embedding`, not a backbone parameter. Historical report labels saying "first backbone parameter" need correction. Initial mean camera is `[4,0,0]`.
- Line 6: v2 initial actual tx/ty are approximately 1 m despite ~0.1–0.2 m hand extent; weighted 2D camera-readout gradient norm 1.517. Pose gradient cosine 3D vs 2D is -0.832.
- Line 8: v2 step 10 has negative weak scales (-0.682 to -0.378) and negative crop-camera z (-103.34 to -57.30). Finite-value checks failed to reject invalid geometry.
- Line 12: v2 step 100 includes scales close to zero and virtual depths up to 5000.76. This explains disappearing projected extent.
- Line 17: v2 step 200 pose-readout gradient norm from 3D is 0.01387 versus 0.00003863 from 2D (about 359x).
- Line 36: v3 step 200 pose-readout norm from 3D is 0.06869 versus 0.007729 from 2D (8.89x); cosine +0.839. Conflict is not universal; 3D dominates this observed pose-head update.
- Backbone gradients are nonzero at the sampled steps. This is gradient evidence, not yet a direct per-backbone weight-change measurement.

Selected correction path: positive weak-scale parameterization + geometric validity reporting, preserve explicit small readout initialization, then independently compare the 2D loss weight under matched initialization/data/steps. Before broad training, run direct MANO self-reconstruction/GT-fit controls to test coordinate/expressivity hypotheses. Measure teacher 2D and real-K fitting against the meeting claims separately.

## Verification Conclusion

Positive camera parameterization and positive-depth assertions pass in six corrected training runs (2,000 total optimizer updates). Actual backbone weight changes are nonzero and checkpoint reload 2D/3D differences are zero. The earlier first-parameter label is corrected in the experiment record.

Exact original pre-reproduction failed after training despite identical initial arrays. The failure is preserved. Two strict deterministic v4 repeats match every saved 2D/3D/camera array exactly; deterministic execution is now the default. Deterministic v4 vs v5 yields 37.632 vs 38.812 px development 2D, so the first non-deterministic apparent reweighting gain is not reliable.

The meeting's two focal interpretations share identical MANO predictions. Real-K initial 2D/depth errors are 18.495 px / 57.817 mm; pseudo-2D translation fitting gives 7.195 px / 64.442 mm (per-image averages). This supports better 2D agreement but does not establish accurate depth or stereo performance.

The NumPy/tensor teacher-output compatibility failure is fixed in run02; both failed and successful evidence remain. Original pre-reproduction comparisons remain explicitly failed, not silently accepted. Full verification and limits: `docs/dyn_hamr_experiment_record.md`, §Runtime debugging; machine-readable `artifacts/debug_experiment_summary.json`.

No removal of debug code/logs is performed. Post-hoc GT fitting remains an oracle, not deployable model accuracy. Next substantive accuracy work requires expanded/diverse data and a separate untouched test; real calibrated stereo is a distinct missing input.
