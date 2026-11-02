# Prospective development protocol: source-level uncertainty decisions

This protocol is fixed before this study's new model fits or selection results.
It follows the audit of branch `uq-conditional-calibration` at `7bf56a6`.
The BTS benchmark, including its anomaly sample, has been examined previously.
These are **development experiments**, never a fresh test or external validation.
The old 1,575-object test split is excluded from fitting, calibration, selection,
and scoring in this study. Feature extraction alone may prepare its inputs.

## Questions and competing designs

1. Does calibrating the maximum score of a complete object trajectory improve
   detection delay or anomaly recovery over Bonferroni at the same lifetime FAR?
2. Does fixed-budget accumulation of genuine conformal rank e-values retain
   useful power at small calibration sizes and sparse anomaly prevalence?
3. Can an observation-support gate increase early automatic classification at
   a certified selective error risk, relative to confidence alone?

The maximum construction, e-merging/e-BH, and learn-then-test certification are
established statistical machinery, **not claimed as new general theory**.
Support gating is inspired by TGLRec; it is not a calibrated probability.
Negative or inconclusive outcomes will be reported without changing endpoints.

## Data and model protocol

- Four fixed looks: days 5, 10, 20, 100 from first recorded detection. Features
  are prefix-causal, rebuilt by the repository's `build_cache.py` inference path.
- One common object-ID universe across horizons. A failed feature extraction or
  inconsistent ID is a hard error; do not silently discard different objects.
- Known development population: the 8,922 train + validation objects only.
  `KFold(5, shuffle=True, random_state=20261102)` assigns object folds without
  using labels. For outer fold k: evaluate k, calibrate (k+1) modulo 5, train on
  the other three folds. No calibration/evaluation object enters fitting or
  early stopping. These fits do not reconstruct Claude's missing nested arrays.
- Use `train_early_cc.hierarchy_nocal`, 2,000 maximum boosting rounds and its
  normal early stopping. Split the proper-training pool 85/15 with stratified
  early stopping, seed 100+k. Retain every model, membership and input hash.
- For each horizon use high top-level energy `-logsumexp(raw margins)` as the
  fixed anomaly score. Location and scale are median and max(IQR,1e-6) on the
  proper-training early-stop rows only. Calibration is not used to fit this
  normalization. Prediction and calibration use exactly the same horizon model.
- The existing 53 anomaly objects are scored only after the design is fixed.
  They are a repeatedly inspected descriptive challenge set, not unseen novelty.
  Report per-fold recovery and object-averaged recovery; never count 5 scores of
  one anomaly as 5 independent discoveries. Do not rank candidate methods on
  these objects or tune any hyperparameter using their outcomes.

## Fixed alarm methods

All ties are conservative. With n calibration objects, upper-tail rank p is
`(1 + count(cal_score >= query_score))/(n+1)`.

- **Naive per-look**: running minimum of each look's p-value; no lifetime claim.
- **Bonferroni**: `min(1, 4 * min_seen p_h)`.
- **Trajectory envelope**: calibration maxima over all four standardized scores,
  queried by the evaluation object's running maximum. The resulting running p
  dominates the terminal split-conformal p pointwise, controlling ever-crossing
  under exchangeability of complete trajectories. This is a strong baseline.
- **E-budget**: rank-to-e calibration with rank weights `r^-2` normalized to
  average one over ranks 1..n+1, then cumulative sum with fixed weight 1/4.
  Never renormalize weights to the number of looks currently available.
- **Trajectory e**: apply the same rank e-calibrator to trajectory-envelope p.

Alarm at lifetime p <= .01 or e >= 100. Report known-object ever-alarm rate,
per-class rates, anomaly recovery, and median first alarm day **among detected
objects**, alongside the detection fraction (delay alone rewards abstention).
Also report final-look e-BH at q=.1 on each fixed batch consisting of its known
evaluation objects and all 53 anomalies; count discoveries and FDP. Do not
interpret a per-object FAR as a queue FDR, or fixed-batch FDR as unlimited
streaming discovery control. Before fitting, report the attainable e ceiling.

## Fixed early classification policy family

At each look, confidence is the probability assigned to the model's hierarchy
label. Multiply by `n_obs/(n_obs+lambda)`; for lambda=0 the gate is one. Stop at
the first eligible look with product >= tau; otherwise abstain.

- tau in {.5,.7,.8,.9,.95}; lambda in {0,5,15}; earliest eligible day in {5,10,20}.
  This is a fixed 45-policy family; no learned threshold outside this grid.
- Each policy's risk is probability of incorrect classification conditional on
  selection, with one decision per object. Certify risk <= .05 using a one-sided
  exact binomial upper bound on the independent calibration objects.
- Allocate delta=.05/5 to each outer fold and delta/(45) within its policy family.
  This gives a simultaneous 95% certificate across all fold-policy statements
  under iid trajectory assumptions. Zero selected calibration objects do not
  certify a policy. If none certify, abstain on all evaluation objects.
- Among certified policies maximize calibration selection fraction; tie-break
  by smallest mean stopping day, then grid index. Report the best confidence-only
  policy (lambda=0) and the best complete-family policy with the **same 45-way
  correction**. A gated winner does not establish gain unless held-out selection
  fraction improves and error/delay tradeoffs also remain acceptable.
- Report evaluation selected fraction, error among selected, mean/median stopping
  time, per-class selected fraction/error, and five fold-specific results. Do not
  treat finite-sample empirical error >.05 as automatic certificate invalidity:
  the certificate concerns population risk with confidence over calibration.

## Interpretation and stopping rules

There is no claim of distribution-free validity under arbitrary survey shift,
adaptive follow-up, changing class mixture, or arbitrary new observation cadence.
The alarm null is the benchmark's known-object **mixture**, not every class
separately. Report rare-class performance rather than concealing this distinction.
Overlapping fits induce dependence across fold outcomes; report fold variability
and raw counts, not a naive independent-fold significance test.

Complete the fixed run, tests, audit and results, including failures. Do not alter
the production defaults on this evidence. A publication-ready efficacy claim
requires a frozen prospective/external object population with provenance and an
agreed deployment policy. Those inputs are not present in the current workspace.
