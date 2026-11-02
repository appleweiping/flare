# Exploratory amendment: learn reliability ranking after the fixed pilot

This amendment is written **after** reading the complete `study-v1` development
results. It is not a preregistration predating those results, and no new test set
is created. The 1,575 original test objects remain excluded from scoring.

## Observation motivating the experiment

The fixed confidence-only policy and the full 45-policy support-gated family chose
identical policies in every fold. They selected 6,725/8,922 objects and made 192
errors (2.855%). The simple n/(n+lambda) support gate offered no selection gain.
Global selective risk hid severe true-class disparities: 18/19 selected SLSN and
5/29 selected TDE were misclassified. This motivates a reliability-ranking
diagnostic, not a claim that global risk control solves rare-object discovery.

## Fixed exploratory alternatives

Reuse the five independently fitted trajectory models and the exact proper-train,
calibration and evaluation memberships of the original protocol. Use **only** the
proper-training early-stopping object IDs (previously used for booster early
stopping) to fit two logistic correctness rankers. Neither calibration labels nor
evaluation labels enter ranker fitting or feature/regularization selection.
This early-stopping reuse can make ranker training optimistic; independent
calibration remains essential and its validity does not require the score model
to be well specified. Use no original test or anomaly labels/predictions here.

Both rankers are StandardScaler + LogisticRegression(C=1, max_iter=2000), with
one row per observed look, each row weight 1/4 so one object has total weight one.
The target is correctness of the existing hierarchy label at that look. Their
sigmoid outputs are treated as decision scores, not guaranteed probabilities.

- Output-only features: six class probabilities, probability of the hierarchy
  label, normalized entropy, top-two margin, and six one-hot hierarchy labels.
- Quality/temporal features: the same outputs plus log(1+n_obs), log(1+day), L1
  change in probabilities since the previous scheduled look, and whether the
  hierarchy label changed. At the first look the two changes are zero. Later
  measurements cannot change an earlier decision.

Each ranker defines 15 policies: score threshold {.5,.7,.8,.9,.95} crossed with
earliest day {5,10,20}. Add these 30 to the original 45 policies for a fixed
75-policy family. Apply one-sided exact binomial upper bounds with tail budget
.05/(5*75), selective risk limit .05. Select the best certified member separately
within (a) original confidence-only, (b) original support family, (c) output-only
ranker, (d) quality/temporal ranker. All four use the same 75-way correction.
Selection and tie breaks are unchanged: highest calibration selection fraction,
then earliest mean decision day, then fixed policy index. Abstain if none certify.

Because the decision to add this family followed inspection of the pilot's
calibration/evaluation outcomes, these reused-sample bounds are **nominal**:
the frozen-family independence assumption is not established for the full
adaptive research process. Do not claim a fresh simultaneous 95% certificate
for this exploratory run. A new independent calibration sample is required to
certify the learned family after it is frozen.

## Report and decision rule

Report the same held-out development selected fraction, error, decision time,
per-true-class selected counts/errors, and all five fold outcomes. Include an
information-matched comparison of the two rankers; an improvement over confidence
alone does not establish a benefit from quality information. Retain fitted gate
parameters, training IDs, scores and hashes. No further hyperparameter search is
allowed in this experiment. Any apparent gain is exploratory and cannot establish
novelty, future survey performance, or class-conditional safety. An external,
temporally separated population and an explicitly rare-class-sensitive objective
are still needed for a publication-level scientific claim.
