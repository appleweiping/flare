# Result-to-claim judgment

Verdict: **PARTIAL — qualified development findings supported; general efficacy and safety claims unsupported.** Integrity status: **warn**, referring to the scope limitations in `EXPERIMENT_AUDIT.md`, not an unresolved numerical defect. Confidence: **medium** for scientific generalization; the reconstructed counts are mechanically supported. Review independence: **same-family**. Acceptance status: **provisional**. Requested reviewer/model/effort and unverified backend identity are recorded in the paired JSON and raw traces.

The deterministic evidence precheck validates the existence of cited raw values only. The semantic judgment below is the fresh reviewer's direct assessment of the code, frozen protocol and raw artifacts, independently of executor summaries. Both experiments reuse the 8,922-object development cohort. There is no paper-writing handoff.

## C1: rank-resolution, rare decisions and resources

**Supported within the observed cohort.** TDE calibration counts of 13, 14, 11, 9 and 11 make TDE permanently admissible for the alpha=.01/.05 deterministic Mondrian envelope. At alpha=.01, multiple permanently admitted labels prevent all singleton decisions. At alpha=.05, the rule emits 67 TDE labels: 5 correct and 62 wrong, giving 7.46% precision / 92.54% error among emitted labels. The relevant source is `rare-budget-v1/results.json` for the selected/correct/wrong counts and `prediction_diagnostics.json` for emitted-label confusion, independently reconstructed from the raw arrays.

At alpha=.10, lifetime Mondrian decisions have 4,721 correct and 382 wrong labels, compared with 7,548 correct and 820 wrong for pooled lifetime sets. Within true SLSN objects, wrong automatic decisions decrease from 96/132 to 7/132; within true TDE objects, from 22/58 to 1/58. These gains in rare true-class protection coincide with a large loss of correct automatic yield. The separate day-100 fixed-look Mondrian control gives 7,120 correct labels after waiting to the deadline; it has a different stopping guarantee.

This supports the revised claim: **“For these frozen four-checkpoint classifiers and reused development splits, class-conditional lifetime sets expose a substantial protection/decision/resource tradeoff, including a structural rank-floor collapse at stringent alpha and poor emitted-label precision despite rare true-class protection.”** It does not establish that finite calibration size explains all losses, that every alternative method suffers the same limit, or that the emitted labels are selectively safe.

## C2: learned continuation utility under matched query budgets

**Partially supported.** The table reports the macro average of six true-class correct-decision rates. Each control value is the largest observed value among all fixed heuristic controls and random seeds in that cell; this is a descriptive comparison envelope, not an evaluation-selected deployable policy. Both frozen macro learner variants are shown, without choosing a winner for deployment. The aggregate-reward variant and every classwise outcome remain in the full result files.

| Protection | Alpha | Extra quota / object | Best control | Control macro | Macro outputs | Macro quality |
|---|---:|---:|---|---:|---:|---:|
| pooled | 0.10 | 0.25 | entropy | 50.716% | 51.387% | 51.311% |
| pooled | 0.10 | 0.5 | entropy | 55.416% | 55.149% | 55.133% |
| pooled | 0.10 | 1 | rare_mass | 64.870% | 63.734% | 63.730% |
| pooled | 0.10 | 2 | round_robin | 66.587% | 66.587% | 66.587% |
| pooled | 0.20 | 0.25 | rare_mass | 59.113% | 58.286% | 58.308% |
| pooled | 0.20 | 0.5 | round_robin | 62.744% | 62.744% | 62.744% |
| pooled | 0.20 | 1 | round_robin | 62.744% | 62.744% | 62.744% |
| pooled | 0.20 | 2 | round_robin | 62.744% | 62.744% | 62.744% |
| mondrian | 0.10 | 0.25 | confidence | 25.022% | 31.377% | 31.407% |
| mondrian | 0.10 | 0.5 | confidence | 27.930% | 33.740% | 33.790% |
| mondrian | 0.10 | 1 | confidence | 32.561% | 36.878% | 36.939% |
| mondrian | 0.10 | 2 | round_robin | 40.859% | 41.753% | 42.066% |
| mondrian | 0.20 | 0.25 | rare_mass | 46.559% | 48.308% | 47.417% |
| mondrian | 0.20 | 0.5 | rare_mass | 50.140% | 52.273% | 52.021% |
| mondrian | 0.20 | 1 | rare_mass | 57.201% | 58.067% | 57.486% |
| mondrian | 0.20 | 2 | round_robin | 63.277% | 63.277% | 63.277% |

All policies tie at b=0 and b=3. The additional ties in the table reflect saturation of all usable paths. The seven nonsaturated Mondrian cells favor both macro learners over the strongest control; three nonsaturated pooled cells favor a control, while pooled alpha=.10 at b=.25 gives a small macro-learner advantage. There is therefore no uniform efficacy result across protection regimes, alpha levels and budgets.

For a concrete matched comparison at Mondrian alpha=.10 and b=1, `macro_quality` reaches macro correct-decision rate 0.36939376594902984 versus 0.3256129820389581 for confidence, a difference of 0.04378078391007173 (4.378 percentage points). It makes 3,385 correct and 179 wrong automatic decisions, versus 3,377 correct and 162 wrong for confidence. The gain is mostly a redistribution of useful decisions across classes, not a large gain in total correct labels, and it is not a reduction in wrong labels. Its foldwise macro differences versus confidence are positive in all five folds, but the folds share training data and are not independent replications.

At Mondrian alpha=.20 and b=.5, `macro_output` gives macro 0.5227273743544855 versus 0.5013955510714853 for rare mass; it has 4,174 correct / 632 wrong labels versus 3,502 correct / 621 wrong. One of five foldwise differences is negative. Conversely, at pooled alpha=.10 and b=.5, `macro_output` is below entropy (0.5514946478320634 versus 0.5541596051654245; exact values are regenerated from the raw results in the evidence record). The pooled losses must accompany the Mondrian gains in any account.

The quality/temporal variant adds observed nobs and previous probability/label changes to a shared schedule basis. Its advantages over macro-output are small at Mondrian alpha=.10, and it loses to macro-output at the nonsaturated Mondrian alpha=.20 cells. This does not support a general claim that quality/temporal features improve continuation allocation.

This supports the revised claim: **“In exploratory playback on this reused cohort, proper-training continuation rankers improve class-balanced correct-decision yield at constrained checkpoint-query budgets for the tested Mondrian lifetime sets, with setting-dependent gains, pooled-calibration counterexamples, and nonuniform error/resource tradeoffs.”** The rankers are reward/cost heuristics, not calibrated utility probabilities. Their feature/reward truth rows are proper-training early-stop objects, while training closure targets depend on separate calibration thresholds; the early-stop scores were already used during classifier development.

## What the results do not support

- General improvement across datasets, models, cadence schedules, protection regimes or future populations.
- Precision/selected-risk certification from true-class unconditional coverage, or a claim that more useful decisions entail fewer wrong decisions.
- Equal photometric-observation cost: budgets match checkpoint-query tokens, while realized nobs and consumed days vary by policy. Queues and integer quotas are separate per outer fold.
- A new conformal theorem or acquisition-method novelty; the protocol explicitly uses established constructions.
- A fresh confirmatory result, a telescope-intervention effect, production readiness or publication readiness.

## Missing evidence and route

For stronger claims, lock the intended policy and operating points before a new, previously uninspected evaluation cohort; estimate uncertainty with a design that respects model/split dependence; enlarge or vary rare-class calibration samples to separate finite-rank from score-quality effects; and evaluate physical follow-up cost/interventions if that becomes the scientific target. Do not tune on these evaluation outcomes and re-label the next analysis confirmatory.

The present route is to record the partial result, retain the negative cells, and narrow the working claim. No additional experiments or writing stage are launched by this review. Machine-readable verdicts, exact input hashes, deterministic precheck and complete reviewer traces accompany this document.
