# Independent numerical audit reproduction

These are byte-identical copies of the three scripts actually used by the fresh reviewer. `manifest.json` records their hashes and the saved verification receipts. They import no experimental calibration, allocation, fitting or summary code. The frozen-booster replay loads LightGBM models and original feature rows without calling the study prediction wrapper. `run_audit.py` is only a relocation/output wrapper; the original scripts are preserved unchanged.

The source root must contain `rare-budget-v1`, `allocation-v1`, `study-v1`, `learned-v1`, `cache`, and `paper_data/data/splits`. Full raw arrays/models are needed; portable result CSV/JSON summaries are insufficient to reconstruct decisions. Python dependencies are numpy, pandas, pyarrow, lightgbm and the project environment used in the recorded execution receipts. The reconstruction scripts write only audit outputs.

Original commands, run from `/workspace/flare` with `PYTHONPYCACHEPREFIX=/workspace/.cache/flare/pycache`, `MPLCONFIGDIR=/workspace/.cache/flare/matplotlib`, `XDG_CACHE_HOME=/workspace/.cache/flare`, and `OPENBLAS_NUM_THREADS=1` where relevant:

```bash
/workspace/.venvs/flare/bin/python .aris/traces/experiment-audit/2026-11-02_rare_budget/reconstruct_block1.py
/workspace/.venvs/flare/bin/python .aris/traces/experiment-audit/2026-11-02_rare_budget/reconstruct_block2.py /workspace/.research/flare-uq/allocation-v1
/workspace/.venvs/flare/bin/python .aris/traces/experiment-audit/2026-11-02_rare_budget/replay_saved_predictions.py
/workspace/.venvs/flare/bin/python -m pytest -q tests/test_uq_rare_class.py tests/test_uq_budget.py tests/test_uq_learned_budget.py
```

To repeat the three numerical audits with an adjustable data location and a fresh output directory:

```bash
/workspace/.venvs/flare/bin/python docs/uq_research/aris/review-stage/audit_scripts/run_audit.py --data-root /workspace/.research/flare-uq --output-root /tmp/flare-independent-audit-repeat
```

The wrapper copies the original scripts into the new directory, changes only their source-root constant in those copies, and executes them there. It refuses to overwrite an existing output directory. No experiment grid or model-training run is launched. The wrapper itself was syntax-checked after packaging; its relocated execution was not rerun as part of this completed audit.

Observed receipts: 2,258 Block 1 checks (65 reconstructed fold-cells), 9,535 Block 2 checks (44 deeper cells and all 77,990 actions within them; sampled deterministic priorities/all random draws), 12 independently recovered weighted-Ridge coefficient fits, 360 exact frozen-booster prediction vectors/nobs, and 101 focused tests passed. No numerical check failed. The checks are finite and sampled; passing them does not establish general scientific validity.

The pinned ARIS `evidence_check.py` also verified 11 cited raw JSON key/value strings. The original numeric-token scan was stopped after over five minutes because its repeated full-prefix slicing scaled poorly on the large result JSON; the replacement used exact key/value strings against the same raw files, with the numerical value retained separately in `claims.json`. This verifies evidence existence only. `precheck_initial_cancelled.json` preserves the failed first invocation and rationale; `precheck_execution.json` records the successful replacement. Independent array and metric reconstruction supplies the arithmetic check.

Complete semantic review prompts/responses and requested-only model provenance remain in the local `.aris/traces/experiment-audit/2026-11-02_rare_budget/` and `.aris/traces/result-to-claim/2026-11-02_rare_budget/` directories. The review is same-family/provisional; no backend model identity was returned by the native spawn tool.
