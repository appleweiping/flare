# Rare-class lifetime decisions under archived checkpoint budgets

Frozen before new real-cohort policy outcomes on 2026-11-02. This is an explicitly posthoc development amendment to the existing study, not a new confirmatory protocol. Scientific design: requested GPT-6-Astra; implementation/execution: requested GPT-6.1-Sol. No paper-writing stage is authorized.

## Problem anchor

Prevent silent wrong automatic labels for rare known transient classes across an object's lifetime, and measure the useful decisions and observation consumption that remain under a limited follow-up budget. The 8,922-object development cohort has already informed research choices. Old test and historical anomalies do not become new validation. Existing twenty classifiers and propertrain/calibration/evaluation memberships stay fixed.

## Claims to test

C1 (diagnostic): replacing pooled protection with true-class lifetime protection can change the achievable decision frontier sharply; finite calibration rank resolution may explain most of the loss. Minimum evidence: per-class calibration counts/infinite thresholds, exact per-class wrong-decision counts, automatic correct decisions, candidate size, unresolved rare objects, and strong fixed-deadline controls.

C2 (exploratory utility): a causal continuation policy trained only on propertrain early-stop trajectories may improve correct-decision yield at identical checkpoint-query budgets relative to uncertainty and static allocation. This is a falsifiable hypothesis, not a promised gain. No method/theorem novelty is claimed: Selection-Safe Conformal Early Exit (2610.20773, Appendix C.3/D.1) and COINS (2609.07112) already contain the core statistical and acquisition ideas.

## Block 1: class-sensitive lifetime sets

Use saved four-look probabilities at days 5,10,20,100. Candidate class score is 1-p_t(c). For class c, calibrate max_t(1-p_t(y)) among calibration objects with y=c. The order index is ceil((n_c+1)(1-alpha)); if it exceeds n_c, use positive infinity. Include ties. At a query prefix, intersect the admitted classes through the current look. Return the first singleton as an automatic label; empty or multiple sets abstain. Later information never changes an issued label.

Compare pooled envelope, Mondrian envelope, pooled Bonferroni, Mondrian Bonferroni, terminal Mondrian, and a separate full-alpha fixed-look Mondrian at EACH deadline. The fixed-look baseline has a guarantee only for its prespecified deadline, not for choosing among deadlines after seeing a query. Bonferroni spends alpha/4 per look. Grid alpha=.01,.05,.10,.20; fixed deadlines=5,10,20,100. Every cell is descriptive; no winner chosen from evaluation labels.

Primary metric is wrong automatic labels divided by ALL objects of each TRUE class. Separately report selected error (wrong/selected), correct automatic count, selection fraction, full scheduled-path true-class exclusion, consumed-prefix exclusion, unresolved retained truth, final set sizes, mean decision day among selected, mean consumed day and cumulative nobs among ALL objects. Fixed-deadline unresolved objects pay the deadline, including empty sets; this deliberate diagnostic cost convention is separate from adaptive operational referral.

Guarantee under within-class exchangeability of whole trajectories and frozen score construction: for each class c, P(any wrong automatic label | Y=c)<=alpha, because a wrong singleton implies exclusion of c on the simultaneous trajectory event. This does NOT guarantee P(wrong | selected,Y=c)<=alpha. Pooled calibration gives a mixture statement only. No guarantee for arbitrary cadence or within-class population change. Threshold estimation uncertainty and adaptively reused development data preclude a new empirical 95% certification claim.

Rank-floor diagnostic is mandatory: class exclusion needs n_c>=ceil(1/alpha)-1. TDE calibration sizes are 13,14,11,9,11 across the five folds, so alpha=.01/.05 envelope cannot exclude TDE for ANY object. If multiple classes have infinite thresholds, singleton decisions are impossible. This is a score-independent finite-sample limit of this deterministic construction, not an impossibility theorem for every method.

## Block 2: exact causal checkpoint-query budget

Initial day5 look for every object is common to all methods and charged separately. One token reveals the next scheduled checkpoint for ONE unresolved object. Global extra budgets B=floor(b*N), b in {0,.25,.5,1,2,3}. Policy sees only each object's currently exposed prefix. It cannot inspect future class sets, probabilities, nobs, or evaluation labels to choose an action. After revealing a new look, record its realized nobs. Singletons close with automatic labels, empty nested sets close with referral, final multiple sets close unresolved. All allocation policies use identical closure rules and identical candidate-set construction. This is archived-checkpoint playback; it is not a telescope-intervention causal evaluation and query tokens are not individual photometric observations.

Controls: round-robin waves (lowest current look, deterministic ID tie), seeded uniform-random active object, current probability entropy, current candidate count, and rare-class probability mass (SLSN+TDE). Learned continuation, if run, uses propertrain early-stop trajectories only and shares available prefix information. Save complete token action ledger, all-object final look/status/prediction, and consumed/unused token counts. Report correct automatic counts and macro class correct-decision rates, wrong-decision rates, candidate utility, and realized nobs cost. Random seeds 0,1,2 assess algorithmic randomness only, not independent dataset uncertainty. No scheduler can overcome a structurally always-included class; a negative utility result ends this mechanism's efficacy claim.

## Block 3: independent audit and scientific decision

A fresh Astra reviewer reads code, raw result paths and this plan, verifies truth provenance, role independence, no future leakage, exact quantile behavior, resource denominators, baseline fairness and claims. Review is same-family/provisional, with requested model metadata distinguished from actual backend telemetry. All new conclusions remain development findings. An interface/tests/reproducible negative frontier is a valid working deliverable; it does not establish publication quality or production readiness.

## Execution and budget

Run toy tests and artifact inventory first; fresh-agent code review before real outcomes. Block1 uses five saved folds with no refit, estimated <2 CPU minutes. Block2 ceiling 30 CPU minutes initially; no GPU required. Persist input membership, parameters, per-object outputs, aggregate counts and reproduction commands under /workspace/.research/flare-uq/rare-budget-v1 and a separate allocation directory. Compact reviewable evidence goes under docs/uq_research/aris/results. Existing source artifacts are never overwritten. Finish all planned cells unless an actual implementation/resource blocker occurs; report unsuccessful hypotheses without changing endpoints after seeing outcomes.
