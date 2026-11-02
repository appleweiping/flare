# Source-level FLARE uncertainty: completed development study

The fixed study ran to completion: 20 newly trained hierarchical classifiers
(five outer splits × four horizons), 8,922 held-out development predictions and
53 previously inspected anomaly objects scored under each refit. No old test
predictions were read. The original 1,575-object test split is excluded.

**The present evidence supports a better audited research framework, not a
claim of a new state-of-the-art detector or a publication-ready efficacy result.**
The e-value and simple support-gate hypotheses did not demonstrate an advantage.
The strongest scientific finding is that pooled safety metrics can conceal
severe failures on rare known classes.

## Protocol, implementation and provenance

The [fixed protocol](uq_anytime_protocol.md) was committed as `ba9e48e` before
these fits. The tested implementation was committed as `f33a67c` before execution.
Every outer split uses three folds for proper training (with an internal early
stopping subset), one for calibration and one for evaluation. All four horizons
use identical object memberships and exactly matching calibration/query models.
The classifier recipe and shipped weights are not changed by this study.

Full results, membership, probability arrays, learned models, score normalization,
input/code/model hashes and a completion checksum are retained at
`/workspace/.research/flare-uq/study-v1/`. Compact reviewable results are in
[uq_research/results](uq_research/results/). The
[independent integrity audit](uq_research/results-audit.md) reloaded all 20 model
pairs and exactly reproduced sampled probabilities, labels and standardized
scores (maximum numerical difference zero). All 8,922 evaluation IDs occur once;
proper training, early stopping, calibration and evaluation membership checks pass.

These are newly reconstructed proper-split experiments. They are not the missing
Claude nested cross-fit/fold-test artifacts, and the e-budget arm does not test
Claude's complete fold-ensemble derandomization proposal.

## Lifetime alarms: power is the limiting factor

All methods below use the same **nominal** lifetime error target .01; they do
not attain the same empirical FAR. Naive per-look alarms have no lifetime control.
Each valid method's reference population is the known-class mixture under
exchangeable complete trajectories, including the observation/context policy.

| Method | Known objects ever flagged / 8,922 | Observed FAR | Anomalies recovered by each refit, out of the same 53 |
|---|---:|---:|---|
| Naive per-look p | 225 | 2.522% | 4, 3, 3, 4, 4 |
| Bonferroni across four looks | 71 | 0.796% | 0, 0, 0, 1, 0 |
| Full-trajectory envelope p | 75 | 0.841% | 0, 0, 0, 1, 1 |
| Fixed-budget rank e-values | 17 | 0.191% | 0, 0, 0, 0, 0 |
| Trajectory rank e-values | 16 | 0.179% | 0, 0, 0, 0, 0 |

Do not describe five refits of 53 anomalies as 265 independent anomalies. The
trajectory/Bonferroni recovery difference is too small, dependent and drawn from
an already inspected challenge set to establish superiority. Both e-BH arms
made **zero discoveries** in every fixed batch at q=.1; zero FDP in this case
does not demonstrate useful detection.

The two trajectory hits are the same object, ZTF17aaapufz, under two refits.
Bonferroni's single hit is ZTF23aaaairl. These are not two distinct trajectory
discoveries versus one independent Bonferroni discovery.

The analytic power check was saved before fitting. At about 1,784 calibration
objects, the chosen exact discrete rank e-calibrator has maximum evidence about
1,085.5. The first of four uniformly weighted looks has at most about 271.4.
An e-BH batch of about 1,838 objects at q=.1 therefore needs at least about 17
simultaneous rejections even at the maximal terminal evidence. This is a real
finite-calibration constraint; it is not a theorem that every possible e-value
method must fail. The trajectory e arm is a monotone transform of trajectory p,
not an independent improvement in ranking.

The fixed energy scorer and median/IQR normalization are also substantive
limitations. They were not tuned after these results, and these results do not
evaluate the full LUNA ensemble or its production lifetime policy.

## Automatic classification: pooled risk masks rare-class failures

Under the fixed finite policy family and proper-split/iid assumptions, exact
one-sided binomial bounds give simultaneous selective-risk certification. The
45-policy family and five outer folds share delta=.05. This is a bound on
misclassification risk **conditional on being selected**, not conformal-set
coverage, anomaly detection or a guarantee for every true class.

Both the confidence-only family and the family including support gates chose
the **same lambda=0 policy in all five folds**. Their held-out results are identical:

- Selected: 6,725 / 8,922 = 75.375%; the remainder abstain.
- Errors among selected: 192 / 6,725 = 2.855%.
- Mean decision day among selected: 40.82 (looks are 5/10/20/100 days).
- Confidence thresholds are .95; earliest eligible days vary between 5 and 20
  according to the independent calibration selection.

| True class | Development objects | Selected | Wrong among selected | Selected error |
|---|---:|---:|---:|---:|
| SN Ia | 6,085 | 5,044 | 24 | 0.476% |
| SN CC | 2,175 | 1,251 | 138 | 11.03% |
| SLSN | 132 | 19 | 18 | 94.74% |
| AGN | 213 | 177 | 3 | 1.695% |
| TDE | 58 | 29 | 5 | 17.24% |
| CV | 259 | 205 | 4 | 1.951% |

These are groups defined by the **true** class, not precision among predictions
of that class. The global risk certificate is compatible with these disparities.
Likewise, trajectory alarms flag 15/58 known TDE objects (25.86%) despite an
aggregate FAR of 0.841%; Bonferroni flags 10/58 (17.24%). This is a mixture-null
limitation, not a coding or label-order failure. Deploying a pooled rule as if it
protected each rare class would be a serious scientific interpretation error.

## Exploratory reliability-ranking follow-up

The [explicit post-result amendment](uq_learned_policy_amendment.md) specifies
one additional experiment: fit output-only and quality/temporal correctness
rankers on proper-training early-stop objects, then compare their first-decision
policies with the original ones under the same 75-way nominal correction.
The reused calibration/evaluation sample makes this follow-up exploratory;
its nominal bounds are not a fresh 95% certification after adaptive research
design. Execution completed for all five folds with saved gate weights, IDs,
scores, coefficients, versions, hashes and a completion checksum.

| Policy family, shared 75-way correction | Selected / 8,922 | Selected fraction | Errors / selected | Mean decision day |
|---|---:|---:|---:|---:|
| Confidence only | 5,367 | 60.15% | 145 / 5,367 = 2.702% | 40.59 |
| Original support family | 6,247 | 70.02% | 166 / 6,247 = 2.657% | 47.54 |
| Learned output-only ranker | 6,183 | 69.30% | 170 / 6,183 = 2.749% | 39.81 |
| Learned quality/temporal ranker | 6,394 | 71.67% | 177 / 6,394 = 2.768% | 42.31 |

Quality/temporal information increases selected fraction by **2.36 percentage
points** relative to the learned output-only control (211 additional decisions,
net), with almost the same aggregate error and decisions **2.50 days later**.
Each fold selects more objects with the quality ranker than with the output-only
ranker. This is an exploratory utility/time tradeoff, not a demonstrated Pareto
improvement, significance claim, or validation of a novel uncertainty method.
The ranker comparison shares the exact same classifier models and training IDs.

Do not advertise the 11.51-point difference against the confidence-only family
as a robust learned-method gain: the stricter 75-way correction leaves that
baseline without a qualifying policy in one fold, whereas the original 45-way
study qualified it. Family size and discrete certification thresholds strongly
influence this comparison. Under the same 75-way correction, the support family
also recovers that fold using a more conservative, later decision policy.

Rare-class safety remains unresolved: the quality/temporal ranker selects 27
true SLSN, of which 19 are wrong (70.37%). The output-only control selects 22,
of which 17 are wrong (77.27%). These small, adaptively inspected counts do not
support a claim of corrected rare-class risk. No production default is changed.
See [the follow-up audit](uq_research/learned-results-audit.md). Full artifacts are
at `/workspace/.research/flare-uq/learned-v1/`; compact results are tracked alongside
the primary study results.

## Research decision

Do not promote any alarm or gate to the production default on these results.
The standalone ingredients have close prior art, reviewed in the
[independent design assessment](uq_research/design-review.md). The defensible next
research question is whether **rare-class-sensitive source decisions** can
improve reliability, discovery utility and decision time under a fixed follow-up
budget, beyond strong information-matched baselines.

Sample size is part of this problem. With zero errors, a one-sided exact 95%
bound below 5% needs at least 59 selected independent calibration examples for
one fixed policy. Correcting 225 fold-policy claims increases this to 164; 2,250
claims need 209. These are optimistic zero-error floors, not power guarantees.
Current per-fold TDE counts are far smaller. Simply relabeling a global bound as
class conditional cannot supply the missing information.

Before an efficacy/publication claim: freeze the final policy and endpoint,
obtain temporally separated objects excluded from all fits and design choices,
record label selection and observation availability, and evaluate under the
actual service cadence. The [mentor handoff](uq_research/mentor_handoff.md)
states the concrete data and scientific decisions required.
