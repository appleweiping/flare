# Experiment integrity audit

Date: 2026-11-02. Verdict: **WARN — numerical and execution checks pass; scientific scope remains exploratory.** Integrity status: **warn**. Evaluation type: **real_gt**, with archived-checkpoint allocation playback. No unresolved implementation or result-integrity blocker was found in this review.

This is a fresh reviewer audit of primary code, dataset labels, saved inputs, calibration arrays, decisions, coefficients and result files. The requested reviewer was `gpt-6-astra` with `ultra` reasoning, returned agent identifier `/root/aris_research_lead/integrity_claim_audit`. The host did not return backend model or effort telemetry. The implementation model was requested as `gpt-6.1-sol`. Identity fields therefore retain requested provenance; `review_independence: same-family`, `acceptance_status: provisional`, and `independence_verified: false`. No cross-family acceptance or publication-readiness determination is made.

## Audited scope and independent reconstruction

The cohort contains 8,922 distinct development objects, evaluated once each across five outer folds: 6,085 SN Ia, 2,175 SN CC, 132 SLSN, 213 AGN, 58 TDE and 259 CV. The classifiers and the four checkpoint horizons (5, 10, 20, 100 days) are fixed. Block 1 contains 480 fold-cells / 96 pooled cells; Block 2 contains 1,320 fold-cells / 264 pooled cells. Random seeds 0, 1, 2 change the allocation algorithm, not the dataset.

The independent audit scripts import no experimental calibration, allocation, training, or summary implementation. They reconstructed 65 Block 1 fold-cells spanning every arm, every alpha level and every fixed deadline. They checked exact thresholds, candidate sets, first singleton labels, resource use, every reported overall/true-class metric and selected pooled confusion matrices. The resulting 2,258 mechanical checks had zero failures.

For Block 2, 9,535 mechanical checks had zero failures. These include the complete cell grid and artifact manifest, all cells' basic budget/ID/correct/wrong accounting, all 20 training-target archives, and 44 deeper cells covering every policy in folds 0 and 3 at pooled alpha=.10 and Mondrian alpha=.20, budget multiplier 1. All 77,990 actions in those deeper cells were reconstructed through closure and final state. Deterministic priority and saved score were independently checked at the first/last 50 actions and every 37th intermediate action; all random draws were reconstructed. Twelve weighted Ridge reward/cost coefficient fits were independently recovered from the raw training features and targets for six fitted models. This is sampled numerical verification, not a claim that every priority in every cell was independently replayed.

The full source/summary manifests were rehashed. Every compact CSV value was matched to its corresponding raw result: Block 1 has 96 overall, 576 true-class and 576 emitted-class rows; Block 2 has 264 overall, 1,584 true-class and 1,584 emitted-class rows. The focused synthetic test suite independently returned **101 passed**. Frozen-booster replay of a separate sample is recorded in the trace alongside these checks.

Raw evidence: `/workspace/.research/flare-uq/rare-budget-v1/` and `/workspace/.research/flare-uq/allocation-v1/`. Independent scripts, numerical receipts and test output: `.aris/traces/experiment-audit/2026-11-02_rare_budget/`.

## A. Ground-truth provenance — PASS

Labels trace to the original `paper_data/data/splits/train.csv` and `val.csv`, not model predictions. Every development membership label agrees with those source split files and each of the four feature caches. Source cache hashes match the original study provenance. Every saved calibration/evaluation label and continuation-training label was independently checked against membership. The 8,922 evaluation IDs cover membership exactly once.

`scripts/uq_anytime/run_study.py:89` selects original train/validation objects; line 104 reads their dataset classes; line 112 saves membership. `run_rare_class.py:32` loads those labels, lines 74–80 enforce exact calibration/evaluation roles and truth, and lines 81–87 validate the frozen model hashes. `run_budget.py:42` checks the saved gate-training archive, exact early-stop IDs and dataset labels. No old-test or anomaly row enters these new outcomes. This audit verifies the local dataset-provided labels; it does not independently adjudicate the underlying astronomical classifications.

## B. Metrics and score normalization — PASS

The candidate score is the explicit probability complement `1-p(c)`. The conformal order statistic is `ceil((n+1)*(1-alpha))`, with positive infinity when the requested rank exceeds the number of calibration scores; ties are retained (`rare_class.py:54`, `rare_class.py:96`). No metric is divided by the model's own maximum or other prediction-derived denominator.

All-object wrong-decision rate, error among selected objects and emitted-label precision remain distinct. `rare_class.py:220` constructs full-path and consumed-prefix exclusions separately; lines 227–251 use the stated denominators. `run_budget.py:67` derives a separate emitted-label confusion matrix/precision and line 88 computes the macro average of six true-class correct-decision rates. Normalized entropy is only an input feature/priority; it is divided by the fixed `log(6)`, not used to inflate an outcome metric. StandardScaler is fitted only on continuation-training rows.

## C. Result existence, completeness and reproducibility — PASS

Both result files match their COMPLETE checksums. Their recorded artifact hashes and copied code hashes match the files. The Block 2 frozen protocol snapshot matches its recorded hash. Both execution receipts record exit code zero, and each fold has the complete frozen grid. Strict JSON represents infinite thresholds with null plus an explicit positive-infinity mask.

Source predictions, four-look membership, calibration labels, training targets, model coefficients, per-object states and action ledgers are saved. The runner never overwrites an existing output directory (`run_rare_class.py:210`, `run_budget.py:233`). Reproduction commands and environment overrides are in the two `execution.json` receipts. Reproducing the full study requires the local source datasets/caches and saved models; the compact CSV files alone cannot regenerate the experiment. Git commit identifiers are supplemented by executed-code hashes and snapshots, so an uncommitted or later code change cannot be silently treated as executed code.

## D. Executed metrics and decision path — PASS

The calibration, nested-set, first-singleton and summary functions are actually called in `run_rare_class.py:164` through line 187. The allocation runner calls the target builder, fitter, allocator, action saver and metric summary in `run_budget.py:143` through line 175. Their saved arrays and metrics were independently reconstructed; no claimed result depends on an uncalled metric function.

Earlier review observed a transient under-development mismatch where the runner named the confidence policy before the scheduler implemented it. The final executed snapshot contains the confidence branch (`budget.py:151`), the associated tests pass, and the completed outputs include that control. This is resolved, not an outstanding defect.

## E. Information boundaries, calibration and baseline fairness — PASS with explicit limits

Training, classifier early-stop, calibration and evaluation memberships are disjoint within every outer fold, constant across horizons, and together cover the cohort. The continuation feature/reward truth rows come only from proper-training early-stop objects. Those objects influenced classifier early stopping, so their classifier scores are not untouched out-of-fold predictions. The continuation targets also use thresholds estimated from separate calibration labels. Both dependencies are disclosed in the frozen protocol and model metadata; neither introduces evaluation truth into training.

`learned_budget.py:21` uses the first future closed state only on training trajectories, weights each eligible object's total fitting contribution to one, and gives no reward to wrong singleton or empty closure. The Ridge hyperparameter and reward/cost clipping are fixed. All variants use the same current-look and log-horizon basis (`learned_budget.py:61`); quality adds only observed nobs and previous probability/label changes. The scorer consumes the exposed prefix (`learned_budget.py:130`).

The allocation engine accepts no evaluation truth. Its ranking uses current probabilities, current set size, or copied revealed-prefix features; a future nobs value is recorded only after spending the token (`budget.py:145`, `budget.py:195`). Full-array validation checks admissibility but does not rank candidates using unrevealed values. All controls and learned policies share thresholds, initial information, closure rules and fold-local token quotas. Confidence, entropy, candidate count, rare mass, round robin and three random seeds are present. No evaluation outcome selects a model, alpha, protection regime or fitted hyperparameter in the runner.

Initial day-5 queries are charged separately. Each additional token advances one unresolved object's next scheduled checkpoint. Each outer fold has its own queue and `floor(b*N_fold)` quota; pooled totals sum those quotas. At b=.5, this is 4,460 tokens, not 4,461. Unused tokens are retained explicitly. Actual cumulative photometric nobs is measured at the consumed checkpoint; it is not the query-token cost and is not summed repeatedly across checkpoints. Block 1 intentionally charges empty unresolved paths through the fixed deadline, whereas Block 2 closes them immediately as referrals. Consequently their nobs totals should not be treated as interchangeable operational costs.

## F. Scope and scientific interpretation — WARN

The analysis is a posthoc development amendment on one already-inspected cohort. It is not a fresh test, prospective resource intervention, arbitrary-cadence evaluation, or external safety certification. The outer-fold models overlap in training data; five fold values and three random seeds are not independent experimental replications. No sampling-uncertainty interval or multiplicity-adjusted efficacy certificate is supplied.

The finite-rank mechanism is real: TDE calibration counts are 13, 14, 11, 9 and 11. At alpha=.05 an envelope-Mondrian TDE candidate cannot be excluded, so any singleton must be TDE. The observed 67 emitted labels contain only 5 correct TDE identifications and 62 wrong labels. True-class unconditional wrong-decision control does not imply precision among labels emitted as TDE. At alpha=.01 multiple always-admitted classes prevent every singleton. These are limits of this deterministic construction at these calibration counts, not an impossibility result for all uncertainty methods.

At alpha=.10, Mondrian lifetime decisions reduce SLSN wrong labels from 96/132 under pooled protection to 7/132, and TDE wrong labels from 22/58 to 1/58, while reducing total correct decisions from 7,548 to 4,721. Fixed-look Mondrian at day 100 yields 7,120 correct decisions at the cost of waiting for that deadline. The matched guarantee is different: a separate fixed-look rule does not provide arbitrary stopping among full-alpha deadlines.

The mathematical wrong-singleton implication is consistent with simultaneous true-class set coverage under the stated whole-trajectory exchangeability assumptions. These reused empirical frequencies cannot establish a deployment risk certificate or selective-risk guarantee. The finite rank floor explains structural collapse at stringent alpha; the audit does not attribute every loss at alpha=.10/.20 to calibration size alone.

## Claim impact and required follow-through

C1 is supported as a descriptive rank/decision/resource finding for this cohort and frozen score. C2 is **partially supported**: the macro learners improve the seven nonsaturated Mondrian operating points beyond the per-cell best control, but lose in several pooled settings and tie once paths saturate. Improved class-balanced correct-decision yield can accompany more wrong labels. Quality features have no consistent advantage over macro-output features. See `CLAIMS_FROM_RESULTS.md` for the complete compact comparison and limitations.

There is no required implementation repair from this audit. Preserve the frozen outputs, negative cells, predicted-label precision, query/nobs distinction and posthoc status in every downstream account. Broader efficacy or safety claims require a new locked evaluation cohort and an explicit uncertainty analysis; physical follow-up savings require a real observation-cost/intervention study. Those are evidence gaps for stronger claims, not unfinished tasks in the present exploratory run.
