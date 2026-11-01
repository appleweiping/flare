# Conditional calibration of the FLARE prediction sets: registered design

Written and committed before any candidate below was scored. The classifier weights, features and
anomaly layer are not changed by this study; only the calibration layer that turns probabilities
into prediction sets, and the review triggers derived from it, are in scope.

## Motivation (measured before this design, on the public benchmark)

The shipped full-curve sets (class-conditional LAC, cross-conformal) on the 1,575 public test
objects: marginal coverage 0.897, but 0.833 for objects with fewer than 15 detections and 0.93 for
30 or more; 2.7% empty sets (all SN Ia / SN CC, argmax correct for 81% of them); with all external
context removed 0.852 marginal, AGN 0.26 and TDE 0.30. The FLARE paper (Sec. 9.2) reports 83%
coverage on the Fritz photometry and states that the sets need their own calibration on that
photometry. ASTRANet (Sasli et al. 2026, Sec. VI H) shows the same failure of class-only Mondrian
across anomaly-score strata and fixes it with an AD-stratified Mondrian.

## Data and predictions

Benchmark train + val (8,922 objects) form the development pool; the 1,575 test objects are read
once, at the end, for the frozen choices. Out-of-fold probabilities come from the repository's own
recipe (`scripts/train_early_cc.hierarchy_nocal`, five StratifiedKFold folds, seed 42) at horizons
5, 10, 20 and 100 days, on prefix-causal features rebuilt from `benchmark/light_curves.tar.gz`.
Every held-out object is predicted under all 16 removal patterns of the four context blocks
(host, Gaia/WISE, PS1 position, M_pseudo). A "thin" model is fitted on light curves that keep a
seeded 30% of the detections (median 9 against 25) and is deployed on the full curves: the
direction and size of the benchmark-to-Fritz density change (26 to 93 detections).

## Groups (fixed now)

For each object at test time, computed from its inputs only:
- `nobs`: tercile of `n_obs_total`, edges from the calibration objects;
- `ctx`: presence of each of the four context blocks (four overlapping binary groups);
- `atyp`: quintile of the classifier energy, -logsumexp of the top-level margins, edges from the
  calibration objects (the classifier-native score of the paper, as in ASTRANet's AD-MCP).
Coverage cells are (true class) x (group value). Cells with fewer than 30 evaluation objects are
reported but excluded from the primary endpoint.

## Scenarios

- `observed`: pattern 0 (the objects as they are);
- `missing`: every evaluation object receives one of the 16 patterns uniformly (seeded);
- `horizons`: the same at 5, 10 and 20 days;
- `shift`: thin-fitted model and thin calibration, evaluated on the full test curves.

Development scores are leave-one-fold-out: each fold's sets are calibrated on the other four
folds' out-of-fold scores, as `scripts/update_coverage.py` does.

## Candidates

- `B0` shipped: class-conditional LAC, threshold per class.
- `B1` B0 with a non-empty rule: an empty set becomes {argmax}.
- `M1` class x group Mondrian (AD-MCP generalised): a threshold per (class, cell) for one
  partition at a time (nobs, ctx pattern, atyp), cells under 10 calibration objects fall back to
  the class threshold.
- `M2` conditional conformal over overlapping groups (Gibbs, Cherian & Candes 2023): per class, a
  linear quantile regression of the score on the indicators of all groups above at level 0.9;
  the fitted quantile is the threshold.
- `M3` M1 / M2 calibrated on dropout-augmented views (each calibration object also under random
  removal patterns), so that the missing-context cells have calibration members.
- `M4` for the shift scenario only: weighted conformal under covariate shift (Tibshirani et al.
  2019), density ratio from a classifier between calibration features and unlabelled deployment
  features of a disjoint half of the deployment objects; halves swapped and pooled.
- `M5` class-conditional APS / RAPS scores in place of LAC (efficiency and empty sets).

Review triggers: the deployed rule sends an object to anomaly review when its credibility is below
0.1 or its own set excludes its predicted class. Candidate `T1` replaces both by one statistic, the
conformal p-value of the predicted class, with its threshold set on known calibration objects for
a 1% per-analysis rate.

## Endpoints and adoption rule

Primary (sets): the maximum under-coverage `U = max_cells max(0, 0.90 - coverage)` over cells
with at least 30 objects, in the `observed` and `missing` scenarios at day 100.
A candidate is adopted only if all of the following hold:
1. `U` falls by at least 0.03 in both scenarios, with a positive lower bound of the paired 95%
   object-bootstrap interval of the reduction (2,000 draws, objects resampled within class);
2. marginal coverage stays within [0.895, 0.915] in `observed`;
3. the mean set size grows by at most 10% and the singleton rate falls by at most 3 points in
   `observed`;
4. no class coverage in `observed` falls below the B0 value by more than 0.02.
Horizons 5, 10, 20 are secondary and must not show a larger U than B0 by more than 0.01.
Shift: `M4` is adopted for the shift recipe only if it moves marginal coverage on the deployed full
curves at least half-way from B0 towards 0.90 with every class within 0.05 of 0.90 or improved.
Triggers: `T1` is adopted if its realised known-class rate on held-out folds is within [0.5%, 2%]
and it recovers at least as many of the 53 out-of-taxonomy objects as the deployed pair of rules
does at its own, higher, rate scaled to equal rate (both read off the same ROC).

### Amendment 1 (before any candidate was scored)

Measuring only the baselines B0/B1 showed that the registered `shift` direction (thin-fitted model
deployed on full curves) raises coverage (0.930), the opposite of the 83% measured on Fritz, so it
cannot test a repair of under-coverage. A second, reverse direction is added and made the primary
shift scenario: `shift_sparse`, the full-curve fold models evaluated on the same held-out objects
with 30% of their detections (B0: marginal 0.773, SN Ia 0.741, TDE 0.69, 7.8% empty sets).
The original direction stays as `shift_dense`, where the goal is to return to 0.90 with smaller
sets. For both, `M4` and every set candidate are scored with the same rule as above: marginal
coverage at least half-way from B0 to 0.90, every class within 0.05 of 0.90 or improved, and the
mean set size reported. Unlabelled deployment features for `M4` are those of the other folds'
objects on the deployment photometry; the evaluated fold never enters the density ratio. Nothing
else in this design changes.

### Amendment 2 (after the first development scoring, before the test objects were read)

An independent audit of the evaluation code and two adversarial reviews per candidate found four
defects, corrected here; every candidate is rescored on the corrected code, the first scoring is
kept as a historical record, and adoption is decided on the corrected scoring only:
1. the weighted quantile gave the test point the weight of one augmented row instead of one object
   (anti-conservative by up to one order statistic for the augmented M3 variants); the test point
   now weighs one object, and every row of a calibration pool weighs 1 / (rows of the same object in
   that pool), so each distinct object in a class pool or in a Mondrian cell weighs one (the first
   rerun weighted cell rows by their global view weight, which shrank the cells fourfold; that
   rerun was discarded before any decision);
2. the endpoint U included an unregistered group family (number of context blocks present); U now
   uses only the registered families (nobs, the four ctx bits, atyp);
3. thinned light curves kept the time origin of the dense curve; the thin caches now restart the
   clock at the first kept detection (and recompute the gaps), and every thin-based array is rebuilt;
4. the augmentation draws of fold 0 reused the seed of the `missing` test patterns; they now use an
   independent stream.

The first scoring showed that no set candidate meets rule 3, and the reviews showed why: exact
coverage in every atypicality stratum lowers the coverage of the most confident objects from about
0.99 to 0.90 by giving them empty sets. One candidate is added POST HOC, designed after the
development results were seen, and labelled as such wherever it is reported:
- `M6`: M1 with partition `atyp` on dropout-augmented calibration (the registered M3 form of M1),
  quintile edges from the calibration objects as observed, plus the non-empty rule of B1 (an empty
  set becomes the hierarchy-rule label). It is scored with rules 1-4 and the shift rule unchanged;
  failing any of them it cannot become the default, and it is reported either way.
- `T1` keeps its registered threshold; its deployed form additionally never flags an object whose
  own 90% set contains its label (threshold min(tau, alpha)), so that at early horizons, where tau
  exceeds alpha, the trigger cannot exceed the deployed rule. This constraint is also post hoc.

The 1,575 test objects then report every adopted and baseline candidate once. An adoption made on
the development pool is not reversed or replaced after the test is read; disagreement is reported.
These remain development results on a previously inspected benchmark, not an external validation.
