# Source-level UQ research

This opt-in study trains separate research models; it does not replace FLARE's
shipped classifiers or production decisions. Read
[the protocol](../../docs/uq_anytime_protocol.md),
[results](../../docs/uq_anytime_results.md), and
[the reading guide](../../docs/uq_research/repository_and_papers_zh.md).

The current extension is described in the
[rare-class and allocation report](../../docs/uq_research/aris/RESULTS.md) and
[public-data feasibility report](../../docs/uq_research/aris/external_feasibility_results.md).
For the pinned scientific workflow, use the tracked
[ARIS installer](setup_aris.py),
[installation instructions](../../docs/uq_research/aris/INSTALLATION.md), and
[workflow record](../../docs/uq_research/aris/WORKFLOW.md).

From the repository root in the prepared cloud environment:

```bash
source /workspace/.venvs/flare/bin/activate
export PYTHONPYCACHEPREFIX=/workspace/.cache/flare/pycache
export MPLCONFIGDIR=/workspace/.cache/flare/matplotlib MPLBACKEND=Agg
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export FLARE_PAPER_DATA=/workspace/.research/flare-uq/paper_data
python scripts/setup_data.py
python scripts/uq_study/build_cache.py "$PWD" "$FLARE_PAPER_DATA" \
  /workspace/.research/flare-uq/cache --tags h5 h10 h20 h100 --workers 4
python scripts/uq_anytime/run_study.py \
  --cache /workspace/.research/flare-uq/cache \
  --out /workspace/.research/flare-uq/study-new --threads 4
python scripts/uq_anytime/summarize.py \
  --study /workspace/.research/flare-uq/study-new \
  --out /workspace/.research/flare-uq/summary-new
python scripts/uq_anytime/learned_policy.py \
  --study /workspace/.research/flare-uq/study-new \
  --cache /workspace/.research/flare-uq/cache \
  --out /workspace/.research/flare-uq/learned-new --threads 2
python -m pytest tests -q -o cache_dir=/workspace/.cache/flare/pytest
```

Use a new output directory for each run. `run_study.py` records protocol, code,
cache and model hashes, saves every fold's models/predictions, and writes a
`COMPLETE` checksum only after all five outer folds finish. It does not score
the old test split. Feature preparation may contain test inputs, so do not
confuse a feature cache with an evaluation result. The old `oof.py` script
prints test accuracy and is deliberately not used by this workflow.

Modules:

- `calibration.py`: conservative rank p/e values, trajectory envelopes,
  fixed-look e budgets, Bonferroni and fixed-family e-BH.
- `policy.py`: one decision per trajectory, fixed support-gated policies,
  exact binomial simultaneous selected-error bounds.
- `run_study.py`: matching proper-training/calibration/evaluation models.
- `learned_policy.py`: separately labelled posthoc reliability-ranking study;
  see the amendment before running it. Its reused-data bounds are nominal.
- `summarize.py`: checksum-verified tables and standalone figures.
- `rare_class.py`: finite-sample pooled/Mondrian lifetime sets and descriptive
  fixed-deadline singleton decisions. `calibrate_sets(...)` fits thresholds;
  `nested_candidate_sets(...)` constructs prefix intersections;
  `first_singleton(...)` returns decisions and consumed looks;
  `summarize_sets(...)` reports overall and true-class metrics.
- `budget.py`: `allocate_query_budget(candidate_sets, probabilities, nobs,
  obj_ids, budget, ...)` returns a `BudgetResult` with all-object `final_look`,
  `predictions`, `status`, `consumed_tokens`, and token `actions`. Policies are
  `round_robin`, `random`, `confidence`, `entropy`, `candidate_count`, `rare_mass`,
  and `custom`. A custom scorer receives only copied, read-only histories in
  `PrefixObservation`; it must return a finite scalar representable as a Python
  float. Higher scores act first, with deterministic ID ties.
- `learned_budget.py`: `training_targets(...)` constructs continuation targets
  on training paths; `fit_continuation_models(...)` fits the frozen aggregate
  output, macro output, and macro quality variants; `continuation_scorer(...)`
  exposes a prefix-only ranking callback. Saved hierarchical labels are
  validated; features use probability argmax and the current look index.
- `run_rare_class.py` and `run_budget.py`: checked saved-fold playback and
  per-object artifacts, with no classifier refitting.
- `summarize_budget.py`: completion/provenance-verified Mondrian allocation
  PNG/PDF figures and an input/output checksum manifest.
- `public_cohort.py`: frozen public-photometry feasibility inventory, retrieval
  and quality checks; it never loads classifier models or predictions. Read the
  [external protocol](../../docs/uq_research/aris/external_feasibility_protocol.md)
  before using this separate workflow. Its CLI is
  `python scripts/uq_anytime/public_cohort.py --help`.

The rare-class and allocation follow-up uses the
[frozen development amendment](../../docs/uq_research/aris/refine-logs/EXPERIMENT_PLAN.md).
Both runners' CLI help is available without reading policy outcomes:

```bash
python scripts/uq_anytime/run_rare_class.py --help
python scripts/uq_anytime/run_budget.py --help
python scripts/uq_anytime/run_rare_class.py \
  --study /workspace/.research/flare-uq/study-v1 --inventory
```

After the amendment's code/scientific review, reproduction uses fresh output
directories (the runners reject an existing output path):

```bash
python scripts/uq_anytime/run_rare_class.py \
  --study /workspace/.research/flare-uq/study-v1 \
  --out /workspace/.research/flare-uq/rare-budget-reproduction
python scripts/uq_anytime/run_budget.py \
  --study /workspace/.research/flare-uq/study-v1 \
  --learned /workspace/.research/flare-uq/learned-v1 \
  --out /workspace/.research/flare-uq/allocation-reproduction
python scripts/uq_anytime/summarize_budget.py \
  --study /workspace/.research/flare-uq/allocation-v1 \
  --out /workspace/.research/flare-uq/budget-figures-reproduction
```

The figure command regenerates the completed allocation comparison without
rerunning allocations. Its x-axis uses actual consumed extra queries; its
six-class macro rate is not rare-only, and random seed ranges are algorithmic
variation rather than confidence intervals. Portable evidence, including full
compressed result JSON, is indexed by the
[portable manifest](../../docs/uq_research/aris/results/portable_manifest.json).
Prefixed `COMPLETE` files verify the decompressed original JSON, while the
manifest verifies the copied/compressed files.

The Block 1 artifact directory is `/workspace/.research/flare-uq/rare-budget-v1`;
Block 2's designated directory is `/workspace/.research/flare-uq/allocation-v1`.
Each completed run writes `results.json`, `provenance.json`, fold inputs and
decisions, code snapshots, and a `COMPLETE` checksum. Allocation additionally
saves continuation models/targets and token action histories. A designated path
alone does not establish that a run completed; check its completion marker and
audit before citing results.

In the allocation scheduler, the initial day-5 checkpoint is available to every
object; its common query count is reported separately. One extra token reveals
one object's next archived checkpoint, regardless of its future `nobs`.
`nobs` records realized cumulative observations after revelation. These are
archive-query budgets, not individual telescope observations or an intervention
on telescope acquisition. Singletons close automatically, empty sets close as
referrals, and ambiguous terminal sets remain unresolved. The earlier
`first_singleton(...)` fixed-deadline diagnostic instead charges unresolved
objects through the deadline, including empty sets.

True-class `wrong_automatic_per_all` divides wrong automatic labels by **all
objects of that true class**. `selected_error` divides wrong labels by selected
objects in that same group. Emitted-label precision instead divides correct
emissions of a label by **all emissions of that label**, across true classes;
it is reported separately by `run_budget.py`. These denominators cannot be
substituted for one another. Continuation scores are ranking heuristics, not
calibrated correctness probabilities. The development objects have already
informed research decisions, so this follow-up remains exploratory and cannot
provide fresh empirical certification or establish a production improvement.

The environment uses the repository's pinned requirements plus pytest, requests
and the tested LUNA install. Model artifacts stay outside Git; compact results
and source/assumption audits are tracked under `docs/uq_research/`.
