# Continuation implementation review and first-pilot interpretation

2026-11-02. Continuation of `/root/aris_research_lead/scientific_review`. Requested native Codex `gpt-6-astra`, `xhigh`; reported model/effort unknown. Same-family/provisional. This reviewer inspected code and raw Block 1 arrays but did not author implementation, rerun tests, or execute Block 2. The separately assigned fresh integrity auditor owns the full independent numerical audit.

## Execution verdict

**READY for the frozen exploratory Block 2 run.** Direct reading of the stable `budget.py`, `learned_budget.py`, `run_budget.py`, corresponding test sources, `reliability_features` dependency and amended experiment plan found no blocking causal, role, target or budget defect. This is a positive pre-run code/design decision. It does not upgrade the scientific REVISE verdict or establish useful scheduler performance.

The initial heuristic panel prioritized high entropy and large sets, which can be weak comparators for correct-singleton yield. The implementation now also includes maximum-current-probability `confidence` priority. This addresses the concrete missing easy-resolution direction before Block 2 results. The output-only and macro-output learned variants provide the principal information/utility-weight ablations; no larger policy stack is needed for this pilot.

## What was verified in the code

`allocate_query_budget` receives full archived arrays for replay and validation but ranks objects and closes paths using revealed entries only. A callback receives copies of one object's current/past candidate sets, probabilities and nobs, with no future backing view or evaluation labels. The copy cannot expose a hidden array tail. The full-path input validator is a precondition check, not an allocation feature. Under this valid-input precondition, changing a still-unrevealed future cannot change the next action.

Every token advances one active object by exactly one look. The queue removes singletons, empty sets and terminal ambiguous objects, preserving the first automatic commitment. Random allocation is uniform over the current active list; deterministic ties use object IDs. The heap caches scores of unchanged objects, which is correct for the stated pure per-object scorer. The current implementation's learned scorer is such a function. A future batch-dependent scorer would require different cache semantics, but no such scorer is used here.

Initial day-5 looks are available to the scheduler; the runner separately charges one initial query per object. Extra token counts equal the sum of final look indices. Unused budget is retained. Actual cumulative nobs is logged at the consumed look and never used to decide whether an unrevealed query is affordable. The budget is therefore an archive-query quota, not a guaranteed photometric-observation or telescope-time quota. Full-budget and zero-extra-budget outcomes must be policy invariant; the toy test source checks both endpoints.

Training targets enumerate active proper-training early-stop prefixes only. They locate the first later singleton/empty closure, or the final look if unresolved. Correct singleton closure earns reward; an earlier wrong singleton cannot be replaced by a later correct label. Cost is the number of additional queries to closure. Each eligible object has total fitting weight one across its prefix rows, and the scaler and Ridge fit both receive those weights. Macro reward N_train/(6*n_train,y) uses class counts from all training objects and the true training label. No evaluation label is used. A zero-positive fold returns a constant-zero reward score rather than a fictitious model gain.

The runner fits each policy using the calibration-defined set path on the early-stop objects, retains explicit model coefficients/training targets and exact role IDs, then supplies evaluation probabilities, nobs and candidate paths to the label-free scheduler. Evaluation truth enters only metric reconstruction. Loaded early-stop IDs and truth are checked against classifier metadata and dataset membership; they are disjoint from calibration/evaluation roles. Separate model/configuration keys are used for both regimes and both alphas. Saved action ledgers and all-object outcomes support independent replay. Pooled metrics sum counts, weight means by their appropriate denominators, and derive predicted-label precision from summed confusion counts.

Test sources cover prefix invariance, action extension as budgets grow, equal token charges, absorbing closures, no evaluation-truth input, fitted-model invariance to evaluation truth/hierarchy labels, object weighting, zero-positive fallback, precision denominators, and a full synthetic policy grid. The executor initially reported 61 scheduler checks and 13 learned-budget checks passing, then 164 related checks after the final common-time-feature refinement; I inspected the test sources but did not independently execute those counts.

## Remaining interpretation limits

- These are expected eventual-closure reward/cost ranking heuristics, not calibrated utility probabilities or a proof of optimal allocation. The revised target is no longer merely immediate-next-look reward; it represents the declared first future closure under continued observation. Its benefit must be established empirically.
- Early-stop trajectories are proper-training objects, but their labels helped select classifier stopping iterations. They are not fully untouched out-of-fold predictions. This may affect policy learning and should be reported; held-out evaluation still measures the resulting frozen policy.
- The continuation model's labels and features come only from early-stop objects, but its targets depend on q fitted from calibration labels. Do not claim that every policy dependency is independent of calibration. This does not invalidate the fixed-path safety argument: the calibrated sets remain unchanged and every possible scheduler commitment is still contained in the same full-path error event. It would matter if the ranker were used to alter the calibrated score or set construction.
- The initial quality variant uniquely received log1p(day), making observational-quality attribution ambiguous. **Resolved before Block 2 outcomes:** the final patch gives every variant the identical log1p(day) and look-index basis. I reread the patch and the explicit common-feature-value test. The quality-only features are now log1p(nobs), previous-look probability L1 change, and previous-look label switch. Any quality-versus-output gain belongs to this three-feature bundle; no individual-feature attribution is established.
- Global query ordering is an archive-inspection sequence, not a calendar-feasible observing schedule. Object-relative day at a consumed checkpoint does not identify when a real survey could request or receive that observation. No additional calendar simulator is needed for this narrow pilot, but a telescope-scheduling or real-time latency claim would require the missing availability process.
- An increase in micro or macro correct commitments at fixed tokens need not improve emitted-label precision or wrong-commitment rate. Report the vector of outcomes. A utility improvement in a poor-precision system is not deployment readiness.

## Block 1: the dual-risk failure is observed

I independently loaded `results.json`, each fold's `inputs.npz`, and each fold's `sets_decisions.npz` under `/workspace/.research/flare-uq/rare-budget-v1/`. For the following cells I reconstructed six-by-six true-row/predicted-column confusion matrices from the per-object predictions and dataset truth. The following are development observations, not certification of a new population.

| Arm, alpha, day 100 | Emitted label | Correct / emitted | Precision |
|---|---|---:|---:|
| Mondrian envelope, .05 | TDE | 5 / 67 | 7.46% |
| Mondrian envelope, .10 | SLSN | 13 / 90 | 14.44% |
| Mondrian envelope, .10 | TDE | 15 / 82 | 18.29% |
| Mondrian envelope, .20 | SLSN | 53 / 305 | 17.38% |
| Mondrian envelope, .20 | TDE | 35 / 101 | 34.65% |
| Mondrian Bonferroni, .20 | TDE | 27 / 411 | 6.57% |

At envelope alpha=.05, all 67 emitted labels are TDE. The 62 false TDE labels comprise 48 true SN Ia, 8 SN CC, 1 SLSN and 5 CV. At the same cell, none of the 58 true TDE objects receives a wrong automatic label: five receive the correct label and 53 abstain. This is exactly why a true-class conditional wrong/all bound does not imply precision among predicted TDEs. A permanently retained TDE candidate can become the last surviving alternative on other classes.

At alpha=.10, the envelope correctly commits 13 SLSN and 15 TDE objects; at alpha=.20, those counts rise to 53 and 35. These are useful numbers to measure, but their emitted-label precisions remain poor. At alpha=.01, the Mondrian envelope makes no commitments, consistent with the multiple-infinite-threshold argument established before outcomes.

The alpha=.05 envelope's overall selected error is 62/67=92.54%, despite its lower nominal true-class lifetime error target. Lower alpha does not order precision of the emitted subset. The Bonferroni alpha=.20 arm similarly emits only TDE and has selected error 384/411=93.43%. These effects are not evidence that the conformal theorem is false; they show that its controlled estimand does not alone meet the operational goal of reliable emitted labels.

The minimum reporting correction is a confusion matrix and precision/counts by predicted label alongside the existing true-class metrics. The executor has now produced complete `prediction_diagnostics.json` and `emitted_label_precision.csv`; these are an appropriate addition, not a new tuned method. Do not silently delete or relabel the failed cells. In user-facing descriptions, call `automatic` a simulated first-singleton commitment state and explicitly state that precision is not certified. No production automatic publication should be inferred from this pilot.

## Reviewer memory update

- Fixed-deadline comparator concern remains resolved.
- Scheduler's missing easy-resolution control is resolved by the added confidence policy before its real outcomes.
- The initial one-step target concern is resolved in the revised eventual-first-closure reward/cost specification, while optimality remains unclaimed.
- The common time-encoding concern is resolved by the final pre-outcome patch and direct reread; remaining quality attribution is to the declared three-feature bundle.
- Rare-class calibration scarcity and the distinction between true-class protection and emitted-label precision are now supported by raw Block 1 evidence. The poor precision blocks any claim of a reliable automatic classifier.
- Block 2 code is reviewed and ready for exploratory execution. Block 2 results and the full final artifact integrity remain unaudited in this receipt.
