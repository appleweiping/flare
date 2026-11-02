# Repository integration checks

Run on 2026-11-02 at 15:47–15:48 UTC in `/workspace/flare`, with base HEAD
`bd3d3499ecc21a18d13970d54d705cd5981b53d4`. The new source files were not yet
committed; the hashes below identify their tested bytes. This receipt records
mechanical integration checks, not scientific acceptance or performance results.
No real-cohort outcomes were run or inspected during these checks.

## Environment and commands

```bash
cd /workspace/flare
export PYTHONPYCACHEPREFIX=/workspace/.cache/flare/pycache
export MPLCONFIGDIR=/workspace/.cache/flare/matplotlib
export XDG_CACHE_HOME=/workspace/.cache/flare
export MPLBACKEND=Agg
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/workspace/.venvs/flare/bin/python -m pytest tests -q \
  -o cache_dir=/workspace/.cache/flare/pytest \
  --junitxml=/workspace/.artifacts/flare/aris-pytest.xml
/workspace/.venvs/flare/bin/python -m pip check
git diff --check
/workspace/.venvs/flare/bin/python scripts/uq_anytime/setup_aris.py --help
/workspace/.venvs/flare/bin/python scripts/uq_anytime/run_rare_class.py --help
/workspace/.venvs/flare/bin/python scripts/uq_anytime/run_budget.py --help
```

Every command exited 0. Pytest reported **229 passed, 1 skipped, 2 warnings in
14.25 s**, with zero failures or errors. The external JUnit file records 230
tests, zero failures, zero errors and one skip.

The skip was `tests/test_atypical.py::AtypicalTests::test_shipped_classifier_modes`:
`atypicality table not exported`. Both warnings were scikit-learn
`ConvergenceWarning` from
`tests/test_analysis.py::AnalysisTests::test_gp_only_interpolates_and_sparse_data_are_explicit`:
the fitted `k2__noise_level` was close to its lower bound `1e-06`.

`pip check` printed `No broken requirements found.` The three CLI checks
displayed the expected options: installer `--project/--source/--dry-run`,
rare-class runner `--study/--out/--inventory`, and allocation runner
`--study/--learned/--out`.

The preserved synthetic scheduler checker was already rerun in the
[engineering audit](BUDGET_ENGINEERING_AUDIT.md); it was not duplicated here.
The public-cohort download CLI was excluded from this initial pass pending its
owner's handoff; the later CLI and receipt checks are recorded separately below.

## Tested source revisions

Paths below are relative to `scripts/uq_anytime/`. SHA-256 values were unchanged
between the start and end of the test suite. Git blob IDs were obtained with
`git hash-object` and do not imply that a commit was created.

| Source | SHA-256 | Git blob ID |
|---|---|---|
| `rare_class.py` | `3260cf01b859d169722cf0227fc2915b18c783a94d7e19af3845e21f1b56b1d5` | `e5eecc7ff579c028593cbacb5e29099edc1f98b1` |
| `run_rare_class.py` | `c9c4f18a7cc13f1ea20653d3d2ba5f89f50e11e7143bf5abfda694af1fa90c21` | `bb75e7b2a6a50bc81e13234b3a70204b71452cec` |
| `budget.py` | `a964ad68db44d0b10d40c3ee5ad4d5dd24e3a50e30722ebe8230ccc7f17c7981` | `e4c1ba0d9b6cbb32c11deb51e9ffda585c0e167c` |
| `learned_budget.py` | `7c6070c14150193c2f71ebe7a5677440b64acf1a6d0f8f5c5f10bfd56475509b` | `4d1c09157059d2c2e304d99c36b06be845f90d84` |
| `run_budget.py` | `8319c46609b42ad060dfef2dc17babf8665c83d4fdbb66f455b90bc24c214593` | `a4e8973c07c32d3ccd52c7fdf66c0a1c4fc10c7d` |
| `setup_aris.py` | `406e75d959cca10fa1813519d02abba38fc0e516beae78e2228548c9fdfd8508` | `0590bc16f200f1879951fd854890849b75d2f32c` |
| `audit_budget.py` | `495e354394567f561c37f9ec57cbcd335de1a6f36d363127b4dd8975b9e2c9d3` | `23107b201eb690375fdf6f95a5e419613a60c02f` |

## Supplemental figure and public-data checks

The core pytest result above remains **229 passed, 1 skipped, 2 warnings**.
These subsequent functional/CLI checks are separate from that suite.

The implementation executor generated the PNG/PDF with the documented
environment and this command, exit 0:

```bash
/workspace/.venvs/flare/bin/python scripts/uq_anytime/summarize_budget.py \
  --study /workspace/.research/flare-uq/allocation-v1 \
  --out docs/uq_research/aris/results
```

The [figure manifest](../results/budget_figure_manifest.json) records the exact
invocation, verified source completion/provenance/code/protocol hashes, plotted
coordinates, class denominators, and output hashes. The PNG was visually
inspected for legibility. The [portable manifest](../results/portable_manifest.json)
was checked against every copied file; both gzip result payloads decompress to
the corresponding prefixed `COMPLETE` checksum, retaining all 480/1,320 fold
cells and all 96/264 aggregate cells. This verified plotting and artifact
reproduction, without rerunning experimental outcomes.

At 15:58 UTC on 2026-11-02 this executor ran the following additional CLI checks
with the environment above; both exited 0:

```bash
/workspace/.venvs/flare/bin/python scripts/uq_anytime/summarize_budget.py --help
/workspace/.venvs/flare/bin/python scripts/uq_anytime/public_cohort.py --help
```

The plot CLI exposed `--study/--out`; the public-data CLI exposed
`--prepare-only`. The public-data execution owner's
[validation receipt](../results/external_stratified_execution_validation.json)
reports a successful cache-only rerun, zero new network requests, 559 unchanged
frozen/raw/object files, and nine synthetic quality-boundary checks. These are
attributed receipt results, not checks independently rerun by this executor.
No public photometry was downloaded again for this integration update. Scope
and interpretation are in the
[external feasibility report](../external_feasibility_results.md).

Supplemental source SHA-256 values:

- `summarize_budget.py`: `bc83e58c2f29bb5893610d1c2611984641d8309204a4cc20cacd03fd7c2c41e4`
- `public_cohort.py`: `a54149a434955e49d0fd312000effa0a99a172001399c57931eca3068892e236`
