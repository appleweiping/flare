# Independent scientific design and pre-run code review

Date: 2026-11-02. Reviewer session: `/root/aris_research_lead/scientific_review`. Requested route: fresh native Codex, `gpt-6-astra`, `xhigh`, no inherited conversation. The host returned a canonical task identifier, not backend model/effort telemetry. Reported model and effort therefore remain unknown; requested identity is recorded separately. `review_independence: same-family`; `acceptance_status: provisional`. This is research review, not paper writing or publication approval.

## Decision

**READY TO RUN the frozen Block 1 exploratory pilot.** I found no blocking correctness defect in the revised `scripts/uq_anytime/rare_class.py` and `run_rare_class.py`. The most consequential missing control in the initial implementation—full-alpha fixed-look Mondrian calibration at each matching deadline—was added and independently reread before this verdict. The decision applies to the six-arm fixed-deadline pilot, not to scheduler implementation that has not yet been reviewed.

**Scientific verdict: REVISE.** The direction is useful as an astronomy development study; it is not an established new statistical method or a publication-ready efficacy result. The immediate contribution is a transparent measurement of how true-class lifetime protection changes automatic-label utility and its finite-calibration feasibility. Adaptive allocation is a small, falsifiable extension, not a way to bypass this feasibility limit.

The frozen problem anchor is preserved: “Prevent silent wrong automatic labels for rare known transient classes across an object's lifetime, and measure the useful decisions and observation consumption that remain under a limited follow-up budget.” The existing development cohort is spent for confirmatory purposes. No new real-cohort policy results were inspected for this review, no experiment was run by this reviewer, and no implementation file was edited by this reviewer.

## What is correct, and exactly what it establishes

For each class c, define the complete-trajectory score M(c)=max_t(1-p_t(c)); calibrate q_c at rank ceil((n_c+1)(1-alpha)) among calibration objects whose true class is c, with infinity when that rank exceeds n_c. Query candidates survive while their running maximum is at most q_c. These candidate sets are nested. Under within-class exchangeability of complete calibration and query trajectories and a score construction fixed outside calibration, the true class survives the entire scheduled path with probability at least 1-alpha, marginally over calibration and a new query within that class. A wrong singleton can occur only after the true class has been excluded. Thus P(any wrong automatic label | Y=c)<=alpha follows directly.

This argument tolerates arbitrary dependence between looks within an object. It is not conditional coverage for each realized calibration sample, selected-label precision, or a guarantee under arbitrary changes in observing policy or within-class population. It controls four declared checkpoints, not every future detection or an unlimited lifetime. Pooled calibration only controls the corresponding mixture population. There is no additional division by the number of classes for the separate true-class conditional statement; Bonferroni across four looks correctly spends alpha/4 within each calibration group.

A scheduler that only decides which of these fixed, already protected paths to reveal next cannot create a wrong singleton outside the same full-path exclusion event. Consequently, the safety argument does not require proving that the scheduler itself is optimal. It does require preserving the fixed score/set construction and treating changes to the actual observations or to the calibrated path as a different problem. Honest prefix causality remains necessary for any empirical utility claim.

Empty nested sets cannot regain a class. Immediate referral therefore saves future checkpoint tokens without changing automatic decisions. Every scheduler comparator must use that same closure rule. The fixed-deadline pilot intentionally continues charging unresolved empty sets to the deadline; that is a documented diagnostic convention. Cost improvements from changing this convention must not be attributed to the learned scheduler.

## Finite-sample feasibility is the central result to measure

A finite threshold requires n_c>=ceil(1/alpha)-1. The inventory supplied to the frozen protocol gives TDE calibration counts 13,14,11,9,11 and SLSN counts 25,20,28,31,28. These counts imply structural conclusions before reading any probabilities:

| Alpha | Envelope minimum n_c | Four-look Bonferroni minimum n_c | Consequence for current TDE calibration |
|---|---:|---:|---|
| .01 | 99 | 399 | Infinite threshold in both constructions |
| .05 | 19 | 79 | Infinite threshold in both constructions |
| .10 | 9 | 39 | Envelope can be finite; Bonferroni is infinite |
| .20 | 4 | 19 | Envelope can be finite; Bonferroni is infinite |

If q_TDE is infinite, TDE belongs to every set for every object. Therefore any singleton must be TDE: automatic non-TDE labels are impossible. If two classes have infinite thresholds, no singleton is possible. At alpha=.01, both TDE and SLSN already have infinite envelope thresholds in every fold, so this deterministic construction cannot make any automatic label. These are properties of this construction and sample size, not an impossibility theorem for all uncertainty methods.

Even a finite q can be near one, leaving the rare class almost never excluded. Conversely, permanently retaining TDE can produce TDE singletons on other true classes; high TDE retention is not precision of a predicted-TDE queue. Report true-class errors, correct automatic decisions, unresolved retained truth, set size, and the full confusion counts if discussing such a queue. Do not choose alpha=.20 after seeing results and then describe it as the operational winner. All prespecified alpha cells, including empty utility, belong in the report.

## Prior art and the narrow contribution

I read the existing online/conditional literature memos and design assessment. I also directly inspected the cached primary text of [Selection-Safe Conformal Early Exit, arXiv:2610.20773](https://arxiv.org/abs/2610.20773), specifically Appendix C.3, D.1, and Proposition 2. It explicitly gives group-specific terminal path quantiles, candidate-wise nested sets, label-dependent group membership evaluated per candidate, the infinity convention, selected-stop coverage, and robustness to group-prior shift with fixed within-group score laws. This rules out a claim that adding Mondrian calibration to the path maximum is a new theorem.

The cached primary abstract and relevant acquisition passages of [COINS, arXiv:2609.07112](https://arxiv.org/abs/2609.07112) already connect any-stage valid nested sets to process utility, stopping time, and acquisition costs. I did not reproduce its full algorithm or prove an equivalence to the proposed scheduler. Its existence nevertheless rules out “anytime validity plus utility-aware acquisition” as sufficient novelty by itself. These are preprints; this review does not infer a peer-reviewed venue from their presence on arXiv.

The defensible thesis is narrower: with fixed astronomical classifiers and the observed rare-class calibration regime, characterize the source-level protection/decision/resource frontier and test whether causal allocation uses a fixed checkpoint budget better than strong same-information controls. A substantial positive astronomy result could be valuable; a negative result that exposes a structural feasibility barrier is still useful. Neither conclusion establishes a top-venue general-method contribution on this historical cohort alone.

## Minimal validation and refinements

1. **Complete the fixed frontier.** Run every frozen arm, alpha, deadline and fold. Compare the newly added fixed-look Mondrian control at the same deadline, and report both actual nobs and all-object stopping day. A fixed deadline is not an equal realized observation budget: early decisions can change consumption. The fixed-look comparator uses the whole alpha at one declared time, so it is a valid, stronger control for a deadline endpoint but cannot be advertised as an adaptive all-deadline method. Initial baseline concern: resolved in code and the frozen plan.

2. **Keep allocation one small experiment.** Use identical protected sets and singleton/empty/terminal closure for every policy, with the declared N initial looks and B extra next-checkpoint tokens. Preserve all-object outcomes and the action ledger. The plan's round-robin, random, entropy, candidate-count and rare-mass controls are an adequate initial heuristic panel. If a learned model is claimed to benefit from observation quality/history, the strongest missing ablation is the same learner with only current classifier outputs and look index; this separates extra information from merely replacing a heuristic with a fitted ranker. Do not add an RL stack. A one-step expected-correct-singleton target is a myopic heuristic and can miss objects that require two advances; describe it literally and test its utility rather than promising optimal allocation.

3. **Audit the completed artifacts and then decide.** A different fresh reviewer should verify raw counts, membership, token accounting and causality after execution. Report micro and per-class/macro correct-decision rates so common SN Ia cannot dominate the claimed rare-class benefit. Record zero selected decisions as undefined selected error, not perfect selective accuracy. Fold variation and random-seed variation are descriptive; overlapping training/calibration folds and repeated randomization do not create independent new cohorts. If no useful automatic-label frontier survives, conclude that these archived data cannot demonstrate the proposed automatic-decision benefit. Candidate retention may remain useful, but it is a different measured benefit and should be named as such.

There is no need for a larger benchmark menu before this pilot. Fresh representative calibration/evaluation trajectories under the deployment cadence are the external prerequisite for a later confirmatory or deployment claim. The current work can finish without them if it keeps its development scope explicit.

## Pre-run code review evidence

The following were checked directly against the stable revised files and test source, rather than inferred from an executor summary:

- `finite_sample_quantile`: exact order statistic, infinity at insufficient rank resolution, no clipping or interpolated quantile, and conservative tie inclusion.
- `calibrate_sets`: correct indexing of the true class at every look, class-specific grouping, pooled mixture distinction, alpha/4 for Bonferroni, and one full-alpha threshold at the selected fixed-look deadline. Earlier fixed-look thresholds are infinite, so no earlier singleton can be issued.
- `nested_candidate_sets` and `first_singleton`: irreversible candidate exclusion, no evaluation truth in construction/selection, first singleton retained even if a later diagnostic set is empty, abstention for empty/multiple sets, and consumption indexed for every object.
- `summarize_sets`: dataset truth rather than hierarchy/argmax labels; wrong/all and wrong/selected denominators remain separate; full-path and consumed-prefix exclusions are different; correct counts and unresolved retained truth are computed correctly; cumulative nobs at the consumed look is counted once.
- Loader and runner: disjoint roles checked by `validate_metadata`; matching memberships across horizons; calibration/evaluation labels checked against dataset membership; anomaly rows excluded; existing model provenance checked; each evaluation ID appears once across folds; fixed-look calibrations are keyed separately by deadline; aggregation sums counts and weights means by their correct denominators; the output directory cannot overwrite an existing run.
- Test source covers finite-rank boundaries, ties and empty classes, exhaustive finite-population held-out assignments, prefix invariance, truth corruption invariance, first-singleton retention, resource accounting, fixed-look deadlines, and synthetic artifact round trips. The executor reported 25 focused and 88 related passing tests. I read their source but did not rerun those tests, so that test execution count is attributed to the executor, not independently observed here.

No blocking defect found within this reviewed scope. This is code review and design assurance, not the post-run integrity audit. Scheduler code and outputs require their separate review.

## ARIS refinement assessment

CALIBRATION: none. No pre-existing human-curated good/bad proposal anchors were supplied or found at the skill's conventional reference location. Scores are subjective diagnostics, not probabilities of acceptance.

| Dimension | Weight | Score / 10 |
|---|---:|---:|
| Problem fidelity | .15 | 9 |
| Method specificity | .25 | 9 |
| Contribution quality | .25 | 4 |
| Frontier leverage | .15 | 8 |
| Feasibility | .10 | 8 |
| Validation focus | .05 | 9 |
| Venue readiness | .05 | 4 |

COMPOSITE: 7.25/10. GAP: there are no curated exemplars to compare; the gap is mainly direct prior-art overlap and absent utility evidence on an untouched cohort, not missing fashionable components. **REVISE** under the research-refine >=9 READY rule. Contribution quality and venue readiness are below seven: the minimum remedy is the narrow empirical framing plus actual matched-budget evidence, and later external validation if a broader claim is sought. That does not block the already specified exploratory run.

Simplification opportunities: keep the classifier and calibration construction fixed; fit at most one small continuation ranker and its information-matched ablation. Modernization opportunities: NONE; a foundation model is not justified by this bottleneck. Drift warning: NONE for the frozen pilot. Reframing zero automatic decisions as demonstrated discovery utility would be drift.

## Reviewer memory update

- Initial suspicion: the terminal-only comparator made early deadlines uninformative. Addressed: fixedlook_mondrian was added separately at every deadline and reread before the pre-run verdict.
- Initial suspicion: low rare-class calibration counts could make every utility comparison trivial. Unresolved by design: the infinity barrier is real and must be reported, not tuned away.
- Initial suspicion: proposed guarantee might be confused with selective risk. Addressed in the frozen plan and code metrics; retain the distinction in every results summary.
- Remaining scientific concern: close prior art prevents a new path-max/Mondrian/acquisition theorem claim. The allowed contribution remains an exploratory astronomy frontier and possible utility improvement.
- Remaining review scope: scheduler causality/budget implementation and raw experimental outputs have not yet been audited here. A later fresh integrity reviewer must cover the completed run.

## Reading receipt

Read the complete pinned Codex-native `research-review/SKILL.md` (165 lines), `research-refine/SKILL.md` (736 lines), and `auto-review-loop/SKILL.md` (548 lines) from `/tmp/aris-source-2927b40/skills/skills-codex/`. Confirmed source revision `2927b40b2ea37774de53763338c5e8f91a7601e6`. Any initially truncated displayed passage was reread in smaller ranges. Also read applicable reviewer-backends, reviewer-routing, reviewer-independence, review-scope-limits, acceptance-gate, review-tracing, taste-calibration, integration-contract helper-resolution section, and output versioning/manifest/language references. Read repository `AGENTS.md` when it became available.

Primary project readings: complete `docs/uq_anytime_results.md`; `docs/uq_research/literature-online.md`, `literature-conditional.md`, `design-review.md`; the frozen `docs/uq_research/aris/refine-logs/EXPERIMENT_PLAN.md`; complete core/runner code and test source, with focused dependencies in `learned_policy.py`. Primary prior-art reading is limited to the sections explicitly identified above, not a claim to have audited both full papers.

Applied stages: independent scientific review, method refinement judgement, and raw-response/memory/assurance obligations needed by the surrounding review loop. Reading `auto-review-loop` does not mean this reviewer executed an entire autonomous experiment loop. No paper-writing, rendering, notification or publication stage was activated.
