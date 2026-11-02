# FLARE UQ research handoff

Start with [the Chinese repository/paper guide](repository_and_papers_zh.md) and
[the completed experimental results](../uq_anytime_results.md).
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

## Reviewable artifacts

[results/](results/) contains raw compact per-fold results, all policy
certification records, object memberships, provenance, evidence ceilings,
summaries and PNG/PDF figures. `manifest.json` records file checksums. Full
model/prediction artifacts are retained outside Git under
`/workspace/.research/flare-uq/{study-v1,learned-v1}/`; the scripts regenerate them.
The figures label fold ranges as descriptive, not independent confidence intervals.

Validation: **128 tests passed, 1 skipped**, with two existing Gaussian-process
convergence warnings. The skip is the pre-existing optional atypicality table
that has not been exported. No tests were disabled to obtain these results.

The old BTS test set and historical anomaly sample are already used scientific
resources, not fresh confirmation. New claims require independent calibration
and prospective/temporally separated evaluation, a defined cadence, and an
agreed rare-class-sensitive loss or review utility. The current learned-ranking
tradeoff is promising enough to discuss, but it does not resolve rare-class safety.
