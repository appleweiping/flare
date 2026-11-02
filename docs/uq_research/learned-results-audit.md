# Independent audit of the learned-reliability experiment

The completed `learned-v1` experiment was audited against its frozen exploratory
amendment, source `study-v1` artifacts, caches, saved classifiers, fitted rankers,
and per-object decisions. The audit performed no tuning, model refitting, or
changes to the experiment's policies. Its outcome is **internally reproducible,
with an exploratory selection gain and a decision-time trade-off**, not a new
formal risk certificate or a demonstrated rare-class safety improvement.

## Integrity and prediction checks

- The learned study's `COMPLETE` marker matches its result-file SHA256:
  `5c5456d8e9e704d79a28c6e4c43add06ea6c0d598755d7b804ec87a21e9ab830`.
- The source-study completion marker and recorded hashes of source results,
  source provenance, original protocol, exploratory amendment, calibration-policy
  code, learned-policy code, and all four caches match their files.
- Per-fold source prediction archives, classifier metadata, all 40 booster
  files, fitted ranker pickles, and gate-training archives match their recorded
  hashes. Pickles were loaded only after their recorded hashes were verified.
- Gate-training IDs are the original proper-training early-stop IDs: 803 or 804
  objects per fold. They are disjoint from classifier-fitting, calibration, and
  evaluation IDs. The four groups partition the 8,922 known development objects
  in each fold; old-test and anomaly IDs do not enter them. Every development
  object is evaluated exactly once over the five folds.
- `_model_predictions` was checked against the actual `HierarchicalFlare`
  implementation for all 20 horizon/fold hierarchies using sampled early-stop
  objects. Probabilities, branch-threshold labels, saved gate-training predictions,
  and observation counts agree exactly. Maximum probability discrepancy: **0.0**.
  The class ordering and hierarchy constants are consistent.
- Independently expanded the reliability features, including previous-look
  probability changes and label switches. They match the implemented features.
  Saved scaler means agree with the gate-training features; feature schemas,
  scaler parameters, and logistic coefficients match the coefficient records.
- Directly evaluating the saved standardized linear predictors and sigmoid
  reproduces every saved calibration/evaluation ranker score exactly. Maximum
  discrepancy: **0.0**. No anomaly rows were used in this calculation.
- Recomputed all 75 policy records, nominal binomial bounds, winners, stopping
  decisions, error counts, and class-specific summaries. They match the saved
  certification and result JSON files exactly.

Machine-readable audit details are retained at
`/workspace/.research/flare-uq/learned-audit-integrity.json`; the audited full
artifacts are under `/workspace/.research/flare-uq/learned-v1`.

## Reproduced held-out development results

All four families use the same 75-policy correction in this experiment.
Selection is at most one automatic classification per object.

| Family | Selected / 8,922 | Selected fraction | Errors / selected | Selective error | Mean decision day among selected |
|---|---:|---:|---:|---:|---:|
| Confidence only | 5,367 | 60.155% | 145 / 5,367 | 2.7017% | 40.587 |
| Original support family | 6,247 | 70.018% | 166 / 6,247 | 2.6573% | 47.536 |
| Output-only learned ranker | 6,183 | 69.301% | 170 / 6,183 | 2.7495% | 39.811 |
| Quality/temporal learned ranker | 6,394 | 71.666% | 177 / 6,394 | 2.7682% | 42.314 |

The relevant comparison for the extra information is quality/temporal versus
output-only: **211 additional selected objects, or 2.365 percentage points of
the development pool**, with mean decision time **2.503 days later**. Empirical
selective error changes from 2.7495% to 2.7682%, and the absolute error count
increases by seven. Neither a fixed-risk superiority claim nor a time-matched
utility improvement follows from these point estimates alone.

The fold-specific net selection differences are `+18, +38, +51, +36, +68`.
The quality method selects 466 objects that the output-only method abstains on,
and abstains on 255 objects selected by output-only. Thus “211 more” is a net
change, not a nested superset of decisions. The five fitted pipelines overlap in
their training data; five positive differences are not five independent
replications and do not supply a confirmatory significance test.

## Why the comparison with confidence alone can mislead

The approximately 11.51-percentage-point advantage over the current confidence
baseline is substantially driven by a certification boundary, not established
incremental discrimination from quality features.

For fold 3, the original pilot's confidence candidate selected 1,309 calibration
objects with 39 errors. Under the pilot's 45-policy correction its nominal upper
bound was **0.0499433**. With the new 75-policy correction, the same candidate's
bound becomes **0.0508119**. No confidence-only candidate then meets the .05
cutoff, so that family correctly abstains on the entire evaluation fold. This
removes 1,334 evaluation decisions compared with the pilot. Folds 1 and 2 also
move their earliest eligible day from 5 to 10, removing another 24 decisions.

Using the common correction is fair within the new comparison, and abstaining
without a certified winner is implemented correctly. However, attributing the
large baseline gap to a better reliability model would conceal this threshold
effect. Report the information-matched learned-ranker comparison alongside all
families and the change in correction. The new quality method also does not
exceed the old pilot's 6,725 selections under that pilot's different correction.

## Rare-class limitations remain

Overall selected error is dominated by common classes. Under the quality/temporal
winner, the true-class counts include:

| True class | Selected | Errors | Error among selected |
|---|---:|---:|---:|
| SN Ia | 4,890 | 23 | 0.47% |
| SN CC | 1,060 | 125 | 11.79% |
| SLSN | 27 | 19 | 70.37% |
| AGN | 163 | 4 | 2.45% |
| TDE | 21 | 3 | 14.29% |
| CV | 233 | 3 | 1.29% |

The output-only ranker has 17 errors among 22 selected SLSN and three errors among
19 selected TDE. These small counts do not establish a rare-class improvement,
and both methods remain unsuitable for a claim of <=5% error in every true
astrophysical class. The global selective-risk objective never promised that.

## Statistical interpretation and audit judgment

The rankers use only the base classifiers' early-stop objects, without calibration
or evaluation rows entering their fit. Reusing early-stop data can make their
training correctness patterns optimistic, but independent calibration would still
be valid for a fixed score model and family under iid trajectory assumptions.

That last qualification matters here: the new feature/policy family was designed
after inspecting this same pilot's calibration and evaluation outcomes. The
exploratory amendment discloses this. Recomputing exact binomial formulas and
applying a 75-way union bound does **not** establish a fresh simultaneous 95%
certificate for the full adaptive research process. The bounds in this run are
nominal, and all gains are exploratory. A newly independent calibration sample
and prospective/external evaluation are needed after freezing the learned family.

No detected implementation error explains the gain or its limitations. This is
a reproducible, modest signal that quality/temporal information may help the
selection-versus-delay trade-off beyond output-only features. It is not sufficient
evidence to change FLARE's production decisions, claim publication-level efficacy,
or claim safe rare-transient automation. Preserve the negative pilot results,
the 75-versus-45 correction effect, subgroup errors, and decision-time trade-off
in any material delivered to collaborators.
