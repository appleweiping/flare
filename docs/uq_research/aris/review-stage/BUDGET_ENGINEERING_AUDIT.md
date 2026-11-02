# Budget scheduler engineering audit

**Outcome:** engineering checks passed after one confirmed numeric-validation defect was repaired by the implementation owner. This is a separate-agent engineering review using GPT-6.1 Sol, not statistical acceptance or cross-family scientific review. No outcome files, performance results, or tuning targets were inspected.

Read `AGENTS.md`, the complete scheduler and its tests, then the owner's changes. Final reviewed SHA-256 values:

- `scripts/uq_anytime/budget.py`: `a964ad68db44d0b10d40c3ee5ad4d5dd24e3a50e30722ebe8230ccc7f17c7981`
- `tests/test_uq_budget.py`: `9b185c35f21b636d6242e60ece65fddeeca6f9c1aedc557a6a5528058369c779`

## Confirmed defect and resolution

The original custom-score validator checked finiteness before converting its scalar to Python `float`. A finite `np.longdouble(np.finfo(np.longdouble).max)` therefore produced an infinite action score and heap priority. The reviewer reproduced this, notified the scientific leader and implementation owner, and made no scheduler edits. The owner now checks finiteness after conversion and added positive/negative overflow regressions. Both regressions executed and passed on this Linux environment. The adjacent confidence baseline was also reviewed: it prioritizes larger current maximum probability, with the existing object-ID tie rule.

## Contract findings

- **Causal isolation:** built-in rankings read only the object's current revealed look. Custom callbacks receive copied, initially read-only prefixes without a backing view into future arrays. Mutating a callback's private copy does not affect scheduler state. Whole-trajectory validation still reads supplied future entries; the causal claim concerns ranking and stopping on valid archived trajectories, not a streaming ingestion interface.
- **Predictions and closure:** labels come only from singleton candidate sets. Probability ties do not force an argmax prediction. Empty sets refer with prediction `-1`; ambiguous final checkpoints become `exhausted`; unresolved objects at budget exhaustion remain `budget`. Singleton, empty, and exhausted paths are removed from future allocation, so an earlier singleton is not revised if later archived sets are empty.
- **Budget exactness:** the free initial checkpoint is index zero. Every action advances exactly one look and costs one archive-query token. `consumed_tokens == len(actions) == sum(final_look)`, with no spending after all paths close. These tokens are not telescope-observation counts or costs.
- **Validation:** shapes, nested binary sets, finite normalized probabilities, finite nonnegative counts, unique nonempty string IDs, integer nonnegative budgets, applicable policy parameters, and finite representable custom scores are checked. Counts are metadata: the API does not require integer or monotone counts. Custom scorers must remain pure functions of their own prefix; cached unchanged-object scores rely on that documented caller contract.
- **Reproducibility:** deterministic score ties use object ID. Seeded random allocation repeats in NumPy 2.3.5, and increasing budget preserves the previous action prefix. This is an environment-specific reproducibility check, not a promise of identical RNG streams across arbitrary NumPy versions.

## Checks actually run

```bash
cd /workspace/flare
PYTHONPYCACHEPREFIX=/workspace/.cache/flare/pycache \
MPLCONFIGDIR=/workspace/.cache/flare/matplotlib \
XDG_CACHE_HOME=/workspace/.cache/flare \
  /workspace/.venvs/flare/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_uq_budget.py
```

Initial revision: **55 passed**. Final repaired revision including confidence: **61 passed in 0.17 s**, exit 0; no skipped tests.

An additional synthetic checker used fixed seed `543802`, 80 small nested-trajectory cases, and a full-recompute reference implementation independent of the scheduler's heap cache. On the final revision it passed **6,192 deterministic reference comparisons**, **3,288 unrevealed-future perturbation checks**, **708 seeded-random budget-prefix checks**, and an explicit tied-probability/empty/singleton case. These checks compare contracts, not method performance. The reproducible checker is [scripts/uq_anytime/audit_budget.py](../../../../scripts/uq_anytime/audit_budget.py), SHA-256 `495e354394567f561c37f9ec57cbcd335de1a6f36d363127b4dd8975b9e2c9d3`. Its bytes are identical to the originally executed `.aris/budget_engineering_checks.py`; no execution-path adaptation was needed. The tracked copy was rerun with the command below and reproduced the same check counts, exit 0:

```bash
cd /workspace/flare
PYTHONPATH="$PWD" \
PYTHONPYCACHEPREFIX=/workspace/.cache/flare/pycache \
  /workspace/.venvs/flare/bin/python scripts/uq_anytime/audit_budget.py
```

## Runtime bounds

For `N` objects, `L` looks, `K` classes, and `B` executed queries, validation and copying cost `O(NLK)` time and memory. Initial ID ordering costs `O(N log N)`. Built-in deterministic allocation costs `O(B(K + log N))` after initialization; random allocation uses constant-time selection/removal plus `O(K)` closure checks per query. No arm scans the whole remaining batch per token.

Custom scoring additionally copies the revealed prefix whenever an object changes. For `r_i` advances of object `i`, this contributes up to `O(K sum_i (r_i + 1)^2)` time, plus the callback's own work; a long checkpoint history can therefore be more expensive than the built-in arms. Action history consumes `O(B)` memory. These are code-derived bounds; this review did not run a performance benchmark.
