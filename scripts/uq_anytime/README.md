# Source-level UQ research

This opt-in study trains separate research models; it does not replace FLARE's
shipped classifiers or production decisions. Read
[the protocol](../../docs/uq_anytime_protocol.md),
[results](../../docs/uq_anytime_results.md), and
[the reading guide](../../docs/uq_research/repository_and_papers_zh.md).

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

The environment uses the repository's pinned requirements plus pytest, requests
and the tested LUNA install. Model artifacts stay outside Git; compact results
and source/assumption audits are tracked under `docs/uq_research/`.
