# FLARE UQ research handoff

Start with [the Chinese repository/paper guide](repository_and_papers_zh.md) and
[the completed experimental results](../uq_anytime_results.md).
For the current extension, read the
[rare-class and allocation results](aris/RESULTS.md) and
[external feasibility results](aris/external_feasibility_results.md).
The deliverable is a reproducible development study and research decision, not
a claim that a top-tier paper or externally validated production improvement is
complete.

## Evidence map

| Material | Scope |
|---|---|
| [Repository and papers](papers-ecosystem.md) | Full-text FLARE/ASTRANet; agents/skybot integration; exact versions |
| [Personal-project transfer](personal-transfer.md) | PEAR/TGLRec mechanisms and limits, with code/commit citations |
| [Existing UQ audit](audit-existing.md) | Recovered branch, calibration/refit issues, queue semantics, missing artifacts |
| [E-values literature](literature-evalues.md) | Primary-source theorem/venue checks, power ceilings, stopping assumptions |
| [Conditional-risk literature](literature-conditional.md) | Coverage vs selected error, LTT, multivalidity, missing data |
| [Online literature](literature-online.md) | Delayed labels, object trajectories, exact prior-art collisions |
| [Independent design review](design-review.md) | Three candidates compared before this study's outcomes |
| [Frozen protocol](../uq_anytime_protocol.md) | Five folds, four looks, fixed comparators and endpoints |
| [Posthoc amendment](../uq_learned_policy_amendment.md) | Learned-ranking follow-up, clearly separated from the first protocol |
| [Primary integrity audit](results-audit.md) | Exact saved-model reproduction and exclusion of old test IDs |
| [Follow-up integrity audit](learned-results-audit.md) | Independent ranker and decision checks |
| [Mentor handoff](mentor_handoff.md) | Unsent English update and prospective validation requirements |
| [Reproduction commands](../../scripts/uq_anytime/README.md) | Code, dependency activation and output locations |
| [Rare-class and query-budget amendment](aris/refine-logs/EXPERIMENT_PLAN.md) | Frozen exploratory lifetime-set and causal archived-checkpoint allocation design |
| [Refined proposal](aris/refine-logs/FINAL_PROPOSAL.md) | Scope, prior-art limits and scientific decision criteria |
| [Follow-up literature check](aris/literature_novelty.md) | Novelty limits for lifetime protection and acquisition policies |
| [Rare-class and allocation results](aris/RESULTS.md) | Completed exploratory playback, utility/protection tradeoffs and evidence limits |
| [External feasibility protocol](aris/external_feasibility_protocol.md) | Frozen public-data selection and original photometry eligibility requirements |
| [External feasibility results](aris/external_feasibility_results.md) | Retrieval/quality assessment only; no classifier-performance validation |
| [Pinned ARIS installation](aris/INSTALLATION.md) | Tracked installer and repeatable project-local scientific setup |
| [Workflow record](aris/WORKFLOW.md) | Executed stages, role adaptation and review provenance |

## Reviewable artifacts

[results/](results/) contains raw compact per-fold results, all policy
certification records, object memberships, provenance, evidence ceilings,
summaries and PNG/PDF figures. `manifest.json` records file checksums. Full
model/prediction artifacts are retained outside Git under
`/workspace/.research/flare-uq/{study-v1,learned-v1}/`; the scripts regenerate them.
The figures label fold ranges as descriptive, not independent confidence intervals.

Earlier study validation: **128 tests passed, 1 skipped**, with two existing Gaussian-process
convergence warnings. The skip is the pre-existing optional atypicality table
that has not been exported. No tests were disabled to obtain these results.

The old BTS test set and historical anomaly sample are already used scientific
resources, not fresh confirmation. New claims require independent calibration
and prospective/temporally separated evaluation, a defined cadence, and an
agreed rare-class-sensitive loss or review utility. The current learned-ranking
tradeoff is promising enough to discuss, but it does not resolve rare-class safety.

## Rare-class and checkpoint-budget follow-up

The source APIs are in [rare_class.py](../../scripts/uq_anytime/rare_class.py),
[budget.py](../../scripts/uq_anytime/budget.py), and
[learned_budget.py](../../scripts/uq_anytime/learned_budget.py). The reproduction
guide above documents the actual runner options and API outputs. The Block 1
artifact directory is `/workspace/.research/flare-uq/rare-budget-v1`; the designated
Block 2 directory is `/workspace/.research/flare-uq/allocation-v1`. Completed runs
have `results.json`, per-fold inputs/decisions, provenance and a `COMPLETE`
checksum. Portable evidence is collected under `aris/results/`; the scientific
report and independent audit must establish what can be concluded from it.

The [result report](aris/RESULTS.md) is the entry point for interpretation.
[Portable evidence](aris/results/portable_manifest.json) retains complete
aggregate frontiers and compressed full fold results; its hashes and the
decompressed original `COMPLETE` checksums serve different purposes. The
[figure script](../../scripts/uq_anytime/summarize_budget.py) regenerates the
[allocation PNG/PDF comparison](aris/results/budget_allocation.png) without
rerunning policy outcomes:

```bash
python scripts/uq_anytime/summarize_budget.py \
  --study /workspace/.research/flare-uq/allocation-v1 \
  --out /workspace/.research/flare-uq/budget-figures-reproduction
```

Run this from the repository root in the prepared environment, using a fresh
output directory. The separate [public-data probe](aris/external_feasibility_results.md)
addresses photometry retrieval and quality eligibility under its frozen protocol;
it supplies no new classifier-performance validation.

One allocation token reveals one next **archived checkpoint** for an unresolved
object. The initial day-5 checkpoint is common to every method and reported
separately; realized cumulative `nobs` is recorded only after a query. This cost
does not model individual photometric observations or an actual change to
telescope acquisition. The scheduler closes singleton labels and empty-set
referrals immediately. The fixed-deadline diagnostic deliberately charges
unresolved objects through the chosen deadline, including empty sets.

The reported rates answer different questions:

| Metric | Denominator |
|---|---|
| True-class wrong automatic rate | All objects of the specified true class |
| True-class selected error | Selected objects of that true class |
| True-class correct automatic rate | All objects of that true class |
| Emitted-label precision | All automatic emissions of that predicted label, across true classes |

Lifetime true-class protection does not itself bound selected error or guarantee
precision for a predicted rare label. Learned continuation uses the original
proper-training early-stop objects; evaluation truth is reserved for descriptive
summaries. All follow-up comparisons reuse development objects and remain
exploratory. Random seeds measure algorithmic randomness, not independent
dataset uncertainty. Review by the requested GPT models is recorded as
same-family/provisional; it does not establish cross-family acceptance, fresh
validation, or publication readiness.
