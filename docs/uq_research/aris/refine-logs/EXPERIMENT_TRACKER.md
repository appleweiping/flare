# Experiment tracker

| Run | Purpose | Scope | Status | Evidence |
|---|---|---|---|---|
| B0 | Artifact roles, toy correctness and causal tests | Existing saved five folds and synthetic checks | DONE | Unit suites; audit traces |
| B1 | Class-sensitive lifetime/deadline frontier | 5 folds × 6 arms × 4 alpha × 4 deadlines | DONE | /workspace/.research/flare-uq/rare-budget-v1/results.json; execution.json |
| B1-D | Emitted-label confusion/precision diagnostic | Derive from saved B1 predictions, no model rerun | DONE | prediction_diagnostics.json; emitted_label_precision.csv |
| B2 | Causal checkpoint allocation | 5 folds × 4 regime/alpha × 6 budgets × 11 policy/seed variants | DONE | /workspace/.research/flare-uq/allocation-v1/results.json; execution.json |
| B3 | Independent integrity and claim audit | Raw B1/B2 arrays, code, source labels and claims | DONE: integrity WARN for scope; claims PARTIAL | ../review-stage/EXPERIMENT_AUDIT.md; CLAIMS_FROM_RESULTS.md; 11,793 reconstruction checks, zero failures |
| E1 | Public external eligibility probe | Up to 64 deterministic IDs per mapped class; 278 total | DONE; eligibility gate failed, 0 eligible | ../external_feasibility_results.md; no model scoring |
