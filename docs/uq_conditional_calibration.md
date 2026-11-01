# Conditional calibration of the prediction sets and the review trigger

A study of how FLARE's 90% prediction sets behave across kinds of objects, of five families of
recalibration, and of the conformal clause of the anomaly-review rule. The design, its three
amendments and every adoption rule were committed before the results they govern
([uq_preregistration.md](uq_preregistration.md)). All numbers below come from the result files in
[uq_study_results/](uq_study_results/) and the code in [scripts/uq_study/](../scripts/uq_study/).
These are development results on the previously inspected BTS benchmark, not an external validation.

## Outcome

| | status |
|---|---|
| **Review trigger** (`review_p_threshold`, `flare.skyportal.review_trigger`) | **adopted**: the conformal clause of `anomaly_review` now fires at a 1% known-class rate per analysis instead of about 8% at day 100 and 3% at day 20 |
| Class-only sets: conditional coverage | **measured failure**: the most atypical fifth of SN Ia is covered about half of the time; AGN and TDE without catalogue context at 0.26-0.30; sparser photometry 0.77 |
| Atypicality-stratified sets (`flare.atypical`, M1/M2/M6) | repair the conditional failure and the sparse-photometry shift, **not adopted** (efficiency rule failed; against a size-matched lower alpha the gain is confined to the energy strata); kept as a research option with no shipped table |
| Proper cross-conformal sets (CC) | met its rule, slightly better than the deployed construction in 5/5 refits at equal size, **not shipped** (a stated deviation, below) |
| Weighted conformal under shift (M4), APS/RAPS (M5) | refuted |

The default prediction sets, the classifiers and the anomaly layer are unchanged.

## 1. What the class-only sets do across objects

The shipped sets threshold `1 - P_c` per class, calibrated cross-conformally on out-of-fold scores.
Their 90% holds on average over each class (out of fold: SN Ia 0.899, SN CC 0.900, SLSN 0.894,
AGN 0.892, TDE 0.914, CV 0.903), but not across how typical an object is. Cells are (true class) x
(group), the groups being the tercile of the number of detections, the presence of each of the four
context blocks, and the quintile of the classifier energy `-logsumexp` of the top-level margins; the
endpoint `U` is the largest shortfall below 0.90 over cells with at least 30 objects.

- **Atypicality.** By energy quintile, out of fold, SN Ia are covered 1.000, 0.999, 0.993, 0.878 and
  0.479 (877 objects in the last; U = 0.421). Under the deployment-faithful nested protocol (Section 3)
  the last cell is covered 0.45-0.54 in each of five refits; on the 1,575 test objects 0.44 (U = 0.464).
  The failure holds under every definition of the stratifier that was tried: energy edges from the
  calibration objects, from the refit's in-sample energies, deciles, or max-P quintiles (0.26-0.61).
- **Missing context.** The classifiers are trained with each context block blanked on 30% of rows,
  but the calibration never conditions on what is missing. With every block removed, test coverage
  falls to 0.852, AGN to 0.26 and TDE to 0.30; with blocks removed at random, out of fold, AGN without
  Gaia/WISE is covered 0.61.
- **Number of detections.** Test objects with fewer than 15 detections are covered 0.833 (4.7% empty
  sets), those with 30 or more 0.93.
- **Sparser photometry.** The full-curve fold models scoring their held-out objects with a seeded 30%
  of the detections (clock restarted at the first kept one): coverage 0.771, SN Ia 0.739, TDE 0.72,
  SLSN 0.76, 8.1% empty sets. This is a simulation in the direction of fainter or younger sources, not the Fritz
  photometry (which is denser, and on which the paper measured 83%).
- **Empty sets.** 2.7% of test sets are empty, all SN Ia or SN CC, whose most probable class is right
  for 81% of them; this is the strict SN Ia threshold (`P_Ia >= 0.85`), not novelty.

## 2. Candidates, development scoring (leave one fold out)

Rules (registered): 1, U falls by at least 0.03 in `observed` and `missing` with a positive lower
bound of the paired object bootstrap; 2, marginal coverage within [0.895, 0.915]; 3, mean set size at
most +10% and singleton rate at most 3 points lower than B0; 4, no class coverage more than 0.02 below
B0. Shift columns give marginal coverage (mean set size). `M6` is post hoc (amendment 2).

| candidate | status | marginal | size | singleton | singleton acc. | empty | U observed [95%] | U missing | sparse shift | dense shift | rules 1/2/3/4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 (shipped) | registered | 0.899 | 1.040 | 0.897 | 0.931 | 0.032 | 0.421 [+0.000, +0.000] | 0.471 | 0.771 (0.99) | 0.932 (1.23) | -/Y/Y/Y |
| B1 (non-empty) | registered | 0.923 | 1.073 | 0.929 | 0.924 | 0.000 | 0.372 [-0.170, +0.000] | 0.330 | 0.836 (1.07) | 0.932 (1.23) | -/-/Y/Y |
| M1 atyp | registered | 0.902 | 1.346 | 0.623 | 0.937 | 0.041 | 0.080 [-0.442, -0.214] | 0.100 | 0.912 (1.53) | 0.892 (1.47) | Y/Y/-/Y |
| M1 atyp, augmented (M3) | registered | 0.912 | 1.164 | 0.792 | 0.953 | 0.041 | 0.059 [-0.467, -0.253] | 0.086 | 0.914 (1.41) | 0.899 (1.75) | Y/Y/-/Y |
| M1 nobs, augmented | registered | 0.909 | 1.160 | 0.807 | 0.951 | 0.030 | 0.362 [-0.194, -0.037] | 0.408 | 0.860 (1.54) | 0.920 (1.24) | Y/Y/-/Y |
| M1 ctx, augmented | registered | 0.898 | 1.218 | 0.767 | 0.930 | 0.021 | 0.442 [-0.101, +0.036] | 0.408 | 0.763 (1.31) | 0.932 (1.45) | -/Y/-/Y |
| M2 cond. conformal | registered | 0.898 | 1.143 | 0.778 | 0.948 | 0.053 | 0.104 [-0.422, -0.184] | 0.294 | 0.918 (1.51) | 0.885 (1.22) | -/Y/-/- |
| M2, augmented (M3) | registered | 0.898 | 1.158 | 0.769 | 0.948 | 0.054 | 0.084 [-0.456, -0.222] | 0.100 | 0.921 (1.52) | 0.889 (1.32) | Y/Y/-/Y |
| M4 weighted | registered | 0.899 | 1.040 | 0.897 | 0.931 | 0.032 | 0.421 [+0.000, +0.000] | 0.471 | 0.892 (2.67) | 0.883 (2.40) | -/Y/Y/Y |
| M5 APS | registered | 0.895 | 1.976 | 0.434 | 0.975 | 0.082 | 0.356 [-0.199, -0.028] | 0.324 | 0.968 (3.05) | 0.861 (1.72) | Y/Y/-/Y |
| M5 RAPS (0.1) | registered | 0.895 | 1.217 | 0.690 | 0.947 | 0.058 | 0.264 [-0.243, -0.094] | 0.264 | 0.920 (1.45) | 0.880 (1.52) | Y/Y/-/Y |
| M6 | post hoc | 0.952 | 1.206 | 0.832 | 0.954 | 0.000 | 0.059 [-0.475, -0.274] | 0.086 | 0.930 (1.42) | 0.906 (1.78) | Y/-/-/Y |

- **What carries the effect is the energy stratum.** Partitions by context pattern or number of
  detections leave U near 0.4; class x energy quintile (the AD-MCP of ASTRANet, Sasli et al. 2026,
  Sec. VI H, with the classifier's own energy) brings it below 0.1, and calibrating on dropout-
  augmented views (each object also under three random context removals, every object weighing one
  per pool) repairs the missing-context cells. The same stratification restores coverage under the
  sparse-photometry shift (0.771 to 0.91-0.93) without seeing any deployment data.
- **No set candidate meets rule 3**, and the reviews show why: exact coverage in every energy stratum
  lowers the coverage of the most confident objects from about 0.99 to 0.90, which costs singletons
  and, without the non-empty rule, turns correct, nearly certain predictions into empty sets. M6 adds
  that rule, which never costs set size but lifts marginal coverage to 0.95 (rule 2).
- **M4 is refuted**: thin and full photometry are almost separable (domain AUC about 0.9), the weights
  hit their clip, and coverage moves towards 0.90 only by adding rare classes to most sets.
- **M5 is refuted**: deterministic APS moves the under-coverage from the most atypical to the most
  typical SN Ia instead of removing it.

## 3. The deployed construction and the nested protocol

The one test read (Section 4) showed B0 at 0.885 and the stratified methods over-covering with larger
sets than in development. The shipped construction calibrates on the out-of-fold scores of five fold
models but predicts with a model refitted on the whole pool, and the development protocol above scored
each fold model on its own held-out fold, so it could not see what the refit does. Amendment 3 adds a
nested protocol that mirrors `scripts/train_full_cc.py`: for each outer fold, an inner five-fold
cross-fit on the other four calibrates, the recipe refitted on those four predicts the outer fold, and
every inner model also predicts it (for proper cross-conformal p-values). D0 is the deployed
construction. Evaluation strata are ranked within the evaluated fold (a label-free, method-independent
definition; with the registered calibration edges the conclusions are unchanged, U of M6 0.044 against
0.048).

| day 100 | scenario | marginal | size | singleton | singleton acc. | empty | U | U change vs D0 [95%] |
|---|---|---|---|---|---|---|---|---|
| D0 (deployed) | observed | 0.903 | 1.076 | 0.879 | 0.931 | 0.025 | 0.433 | reference |
| D0 + non-empty | observed | 0.923 | 1.101 | 0.903 | 0.927 | 0.000 | 0.433 | +0.000 [-0.157, +0.000] |
| M6 on the refit | observed | 0.952 | 1.238 | 0.802 | 0.953 | 0.000 | 0.048 | -0.385 [-0.500, -0.287] |
| CC | observed | 0.912 | 1.075 | 0.889 | 0.933 | 0.020 | 0.467 | +0.033 [-0.079, +0.111] |
| CC6 | observed | 0.963 | 1.285 | 0.770 | 0.965 | 0.000 | 0.067 | -0.367 [-0.471, -0.237] |
| D0 (deployed) | missing | 0.893 | 1.084 | 0.864 | 0.926 | 0.028 | 0.452 | reference |
| M6 on the refit | missing | 0.949 | 1.265 | 0.789 | 0.950 | 0.000 | 0.094 | -0.358 [-0.401, -0.208] |
| CC | missing | 0.902 | 1.098 | 0.871 | 0.925 | 0.019 | 0.409 | -0.043 [-0.093, +0.089] |
| CC6 | missing | 0.955 | 1.310 | 0.758 | 0.955 | 0.000 | 0.100 | -0.352 [-0.408, -0.204] |

| day 20 | scenario | marginal | size | singleton | singleton acc. | U |
|---|---|---|---|---|---|---|
| D0 (deployed) | observed | 0.901 | 1.330 | 0.713 | 0.914 | 0.344 |
| M6 on the refit | observed | 0.922 | 1.783 | 0.468 | 0.927 | 0.012 |
| CC | observed | 0.913 | 1.342 | 0.702 | 0.930 | 0.400 |
| CC6 | observed | 0.953 | 1.784 | 0.461 | 0.959 | 0.036 |

Per refit at day 100 (observed), marginal coverage of D0 is 0.892-0.918 and that of CC 0.904-0.920;
the SN Ia of the top energy quintile are covered 0.45-0.54 by D0 and 0.82-0.93 by M6. The refit-to-
calibration energy offset is random in sign (20th-percentile shifts +0.01, -0.23, +0.35, -0.33, -0.26),
not the systematic offset that the single test refit showed, and the five fold models themselves differ
in median energy from -1.57 to -2.54: the energy has no common scale across models.

An adversarial review then compared M6 with D0 run at a lower alpha chosen only so that its mean set
size matches (alpha 0.055 at day 100, 0.0525 at day 20). At day 100, M6 still lowers U (-0.152,
[-0.238, -0.065]) and raises the most atypical SN Ia by 0.12-0.20 in every refit, AGN and CV gain too,
but the advantage lies in the energy family it is stratified on (on the detection-count and context
families the size-matched D0 has U 0.018 against 0.048), SN CC and SLSN cells lose coverage, and the
bootstrap fails rule 1 in `missing`. At day 20 M6 is no more efficient than the size-matched D0. This
is why the atypicality sets are not offered as an option in the package, although the module stays for
the evaluation the paper asks for on Fritz photometry.

**Deviation from amendment 3.** CC met its numerical rule (marginal 0.912 / 0.902, class coverage and
efficiency equal to D0's) and is a little better than D0 in 5 of 5 refits at equal size (SN Ia top
quintile 0.539 against 0.484), which the max-over-cells U hides (it is set by a 30-object AGN cell where
the two differ by one object). Amendment 3 says CC then replaces the deployed calibration. It is not
shipped in this change: it needs the five fold models at inference, which the shipped training run did
not keep, and it would change what the reported p-values, credibility and confidence mean relative to
the review trigger. The gain is small and post-test; the decision is left to the maintainers.

## 4. The one test read (1,575 objects)

Calibration on all development out-of-fold rows, predictions by the refit on the whole development pool.
For the candidates designed afterwards (Section 3) these numbers are contaminated, descriptive checks.

| candidate | marginal | size | singleton | empty | U observed | U missing | day 20 / 10 / 5 U | sparse shift | dense shift |
|---|---|---|---|---|---|---|---|---|---|
| B0 (shipped) | 0.885 | 1.029 | 0.901 | 0.037 | 0.464 | 0.496 | 0.264 / 0.132 / 0.062 | 0.771 (0.99) | 0.928 (1.23) |
| B1 | 0.916 | 1.065 | 0.938 | 0.000 | 0.285 | 0.304 | 0.259 / 0.132 / 0.062 | 0.839 (1.07) | 0.928 (1.23) |
| M1 atyp | 0.929 | 1.340 | 0.657 | 0.020 | 0.013 | 0.037 | 0.158 / 0.084 / 0.137 | 0.908 (1.41) | 0.911 (1.47) |
| M1 atyp, augmented | 0.928 | 1.379 | 0.633 | 0.012 | 0.033 | 0.020 | 0.061 / 0.009 / 0.058 | 0.907 (1.48) | 0.908 (1.56) |
| M2, augmented | 0.926 | 1.180 | 0.793 | 0.032 | 0.005 | 0.015 | 0.133 / 0.044 / 0.069 | 0.921 (1.54) | 0.897 (1.26) |
| M6 | 0.941 | 1.385 | 0.650 | 0.000 | 0.033 | 0.020 | 0.053 / 0.003 / 0.058 | 0.916 (1.49) | 0.942 (1.55) |

The shipped weights with their own calibration cover these test objects 0.897 (the audit that started
this study); this refit covers 0.885, and its energies sit about 0.25 above the fold models' at every
quantile, an instance of the refit-to-refit variability measured in Section 3 (0.892-0.918).

## 5. The review trigger

`rule_verdict` sent an object to `anomaly_review` when its credibility was below 0.1 or its own set
excluded its predicted class. On every held-out object both clauses together are exactly "the conformal
p-value of the predicted class is at most alpha = 0.1" (a credibility below 0.1 implies an empty set),
a review rate that nothing calibrates: 7.96% of known objects per analysis at day 100 out of fold, 41% of
them empty sets, 3.1% at day 20, 0.15% at day 10 and 0.07% at day 5, alongside a consensus alarm
calibrated to 1% per analysis and to a lifetime false-alarm rate.

The adopted trigger keeps the statistic and sets its threshold for a 1% known-class rate: the largest
`tau` whose leave-one-out rate on the out-of-fold objects is at most 1% (`tau` = 0.0319 at day 100), and
never above alpha, so it never flags a label that its own set contains (`min(tau, alpha)`; at days 5 and
10, where `tau` exceeds alpha, the deployed rule is unchanged).

| | known-class rate, out of fold | on test | out-of-taxonomy objects flagged (of 53) |
|---|---|---|---|
| deployed rule, day 100 | 7.96% | 8.83% | 9 |
| deployed rule scaled to the trigger's rate | 0.98% | | 1.1 |
| **trigger, day 100** | **0.98%** | **1.78%** | **4** |
| deployed rule / trigger, day 20 | 3.08% / 0.98% | 2.86% / 0.89% | 7 / 4 |

At equal rate the trigger catches about 3.5 times as many out-of-taxonomy objects as a random thinning
of the deployed rule. In absolute terms it catches fewer (4 against 9, all novae) at an eighth of the
review load; at day 100 it fires only on objects predicted SN Ia (for the other classes an argmax label
cannot reach so low a p-value), and its rate is about three times higher for the sparsest light curves.
The count of 4 is small. The threshold ships for the full-curve model only; the early models keep the
deployed rule until a threshold is exported from their own training pool (day 20 would bind at 0.080).

## Limitations

The benchmark has been inspected repeatedly, here and in the paper; the test split was read once and is
spent. No fresh labelled population was reachable from this environment (TNS returns 403 without bot
credentials, Fritz requires authentication, ALeRCE has no spectroscopic labels). The photometry shifts are
simulations on the same objects; the shift that matters for the service is the Fritz photometry, on which
the sets should be recalibrated and these candidates rescored (the paper, Sec. 9.2). Rare-class cells are
small (TDE 68, SLSN 156 objects in all), and the object bootstrap holds the fitted thresholds fixed, so
its intervals understate the variability of thresholds estimated from 10-30 objects. The shipped
`review_p_threshold` comes from a cross-fit that reproduces the recipe of the shipped full-curve weights on
the same pool, not from the original fit; the early models were fitted on a pool that also holds Fritz
objects, so no threshold is exported for them here.

## Reproduction

`scripts/uq_study/README.md` lists the commands in order (feature caches, cross-fits, sealed test
predictions, scoring, decisions, nested protocol, report tables). `scripts/export_review_threshold.py`
writes `review_p_threshold` into a model directory's `conformal.json` (and, with `--atypical`, the
research table). Tests: `python -m unittest tests.test_atypical`.
