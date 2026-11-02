# Independent post-run audit: source-level UQ study v1

Audited `/workspace/.research/flare-uq/study-v1` after all five folds completed. The frozen protocol is `docs/uq_anytime_protocol.md`; recorded experiment commit is `f33a67c8137843f09cc09f55a14d53d72f191a5f`. No experiment code, production module, decision rule, model, or result file was edited by this audit. The machine-readable audit output is `/workspace/.research/flare-uq/audit-results-integrity.json`.

## Integrity and reproduction: passed

- `COMPLETE` matches the SHA256 of `results.json`: `4574c70a3fbce260ce87e184ee10a177bf86db8305d2fe6bc9d1493bf06fab45`.
- Protocol, all three recorded Python source files, and all four feature-cache hashes match provenance. All recorded top/branch model hashes match their files.
- All five fold result files equal their entries in `results.json`.
- Exactly 8,922 distinct development objects appear once each in evaluation; calibration is the next fixed fold. Calibration, evaluation, proper training, and early stopping have the required disjointness. Training plus ES is exactly the three permitted folds. The 1,575 old-test IDs do not appear in any fit/calibration/evaluation/scored-anomaly membership. All 53 descriptive anomalies are accounted for separately.
- Reloaded all **20 hierarchy models** (four horizons times five folds; 40 booster files). For every hierarchy, six rows spanning calibration, evaluation, and anomaly data reproduce saved probabilities, hierarchy labels, observation counts, and standardized energy scores. Maximum absolute probability and score errors are both **0.0**.
- Recomputed median and IQR from every model's saved ES IDs and feature order. All 20 normalizers reproduce exactly, with maximum absolute error **0.0**.
- Class-name order is identical between `evaluate_unified.SIX`, the saved y arrays, and `HierarchicalFlare`. This rules out a label-permutation explanation of the rare-class failures.
- Independently reconstructed upper-tail rank counts by direct comparisons, trajectory-envelope ranks, budgeted e-values, and every saved alarm path. All decisions and reported counts agree.
- Recomputed every policy's certification and winning-policy selection from saved calibration predictions; the JSON agrees exactly. Evaluation decisions and errors agree with saved results.
- No old-test prediction file was opened or generated during this audit. Cache metadata was used to verify exclusion; no model scoring used old-test rows.

## Alarm outcomes verified

All known objects are evaluated once. At the fixed nominal lifetime threshold .01:

| Method | Known ever-alarm count | Known rate | Anomaly detection occurrences across 5 fits | Mean recovery over the 53-object challenge set |
|---|---:|---:|---:|---:|
| Naive repeated per-look rank | 225 / 8,922 | 2.5219% | 18 / 265 | 6.7925% |
| Bonferroni | 71 / 8,922 | 0.7958% | 1 / 265 | 0.3774% |
| Trajectory envelope | 75 / 8,922 | 0.8406% | 2 / 265 | 0.7547% |
| Fixed-look e-budget | 17 / 8,922 | 0.1905% | 0 / 265 | 0% |
| Trajectory e-value | 16 / 8,922 | 0.1793% | 0 / 265 | 0% |

The 265 denominator is 53 objects replayed through five fitted pipelines, **not 265 independent anomalies**. Trajectory envelope's two occurrences are the same object, `ZTF17aaapufz`, detected in two folds. Bonferroni's one occurrence is `ZTF23aaaairl` in one fold. The naive method detects eight distinct anomaly objects across the fits. These tiny challenge-set counts do not establish superiority of trajectory envelope over Bonferroni; they establish weak power at the chosen operating points.

Final-look e-BH yields **zero discoveries in every fold for both e-value methods**. FDP=0 is therefore vacuous as an efficacy result, not evidence of useful discovery. This matches the prespecified finite-rank evidence ceiling concern. Trajectory-e is a conservative rank transformation of trajectory-p and its threshold represents a more stringent empirical tail; it is not a different ranking method at an empirically matched FAR.

The naive repeated-look method inflates the source-level false-alarm rate over the per-look nominal target, while both multiplicity-corrected p methods reduce it below 1% in this development run. These are empirical development rates, not a distribution-shift guarantee or external validation. Correlated fitted folds prevent treating five results as independent replications.

## Important rare-class failure hidden by global summaries

The known-null alarm guarantee is for the sampled **known-class mixture**, not each known class. The subgroup outcomes make that distinction operationally important:

| Method | Known TDE alarms | TDE rate |
|---|---:|---:|
| Naive | 29 / 58 | 50.00% |
| Bonferroni | 10 / 58 | 17.24% |
| Trajectory envelope | 15 / 58 | 25.86% |
| E-budget | 5 / 58 | 8.62% |

Thus the trajectory method's 0.84% global FAR can coexist with alarms on roughly a quarter of known TDEs. This is not a model/label-order implementation bug and does not contradict the mixture-level statement. It limits any claim that normal members of every astrophysical class are protected at 1%.

## Early automatic classification outcomes verified

All five support-family winners are confidence-only (`support_lambda=0`) and are exactly the baseline-family winners. All choose tau=.95; earliest eligible days are `[20, 5, 5, 20, 20]`. The support-gate hypothesis has no demonstrated improvement in this fixed experiment.

The selected policies classify 6,725 / 8,922 objects automatically, with 192 errors: **2.8550% pooled selective error**. This is an overall mixture result. The selected-true-class confusion matrix is reproduced in `audit-results-integrity.json`.

- True SLSN: 19 selected of 132; **18 / 19 selected are wrong** (94.74%). They are labelled SN Ia six times, SN CC eleven times, CV once; only one is correctly called SLSN.
- True TDE: 29 selected, five wrong (17.24%).
- True SN CC: 1,251 selected, 138 wrong (11.03%).
- True SN Ia: 5,044 selected, 24 wrong (0.476%).

The overall certificate controls population error conditional on selection under iid trajectories; it does not promise true-class-conditional error. The majority SN Ia population dominates the selected sample. This is a scientifically material failure of the global objective for rare-transient use, even though the mixture certificate and code are consistent. Report this prominently and do not deploy the automatic-classification policy on the basis of its global 2.86% figure.

These subgroup findings can motivate the next explicitly exploratory or prospectively frozen study. They must not be used to change this run's winner or retune its constraints while preserving the original preregistered label.

## Test-status check

The preserved full-run log reports **112 passed, 1 skipped, 2 warnings**. Independently rerunning only `tests/test_atypical.py` produced **6 passed, 1 skipped**. The skip is precisely `tests/test_atypical.py:55`, reason **“atypicality table not exported”**; the research-only optional table is not shipped. It is not a silently skipped failed core calibration test. No unrelated test suites were rerun by this audit.

## Judgment

The run is internally reproducible and the reported weak/negative findings are supported. There is no detected data-ID, label-order, model-loading, score-normalization, or decision-count bug explaining the outcomes. The valid deliverable is an audited research baseline, sound research calibration utilities, and clear scientific limitations. It is not a demonstrated performance improvement suitable for changing FLARE defaults, and it is not yet a publication-level efficacy result.
