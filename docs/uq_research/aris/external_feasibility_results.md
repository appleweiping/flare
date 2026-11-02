# Public external-cohort feasibility results

**The frozen preflight threshold was not met: 0 of 278 selected objects were eligible, including 0 of 12 SLSN and 0 of 10 TDE.** No external model evaluation or calibration was attempted. The required threshold was at least 100 eligible objects overall and at least 10 each SLSN/TDE.

The [protocol](external_feasibility_protocol.md) was frozen at 2026-11-02 15:47:35 UTC, before selection was frozen at 15:49:16 and before the first cohort request at 15:50:08. Protocol SHA-256: `d2efff7f134ffa2577b3eeed41e43682e1d4dd8b5137da89bbbf654664328cd2`. The initial cohort assessment completed at 15:52:08 UTC. The script and frozen selection are retained; no failed object was replaced and no cut was changed.

| Class | Mapped pool outside exclusions | Selected by SHA-256 order | Valid responses | Eligible | Median first-100-day detections |
|---|---:|---:|---:|---:|---:|
| SN_Ia | 1,175 | 64 | 63 | 0 | 5 |
| SN_CC | 420 | 64 | 64 | 0 | 7 |
| SLSN | 12 | 12 | 12 | 0 | 19 |
| AGN | 153 | 64 | 64 | 0 | 4.5 |
| TDE | 10 | 10 | 10 | 0 | 2.5 |
| CV | 382 | 64 | 64 | 0 | 4 |

All 277 valid photometry responses had mature windows relative to 2026-11-02 00:00 UTC, and all failed the original count/band quality predicate. The overlapping failure flags were 215 objects with fewer than 8 total detections, 138 with fewer than 2 g-band detections, and 114 with fewer than 2 r-band detections. SLSN often had many detections but inadequate band coverage: its median first-window counts were 0 g and 12 r. TDE had a median of 2.5 total detections. One SN_Ia endpoint returned HTTP 404. Nothing was rescued by changing the window origin, removing band requirements, or adding later outbursts.

A post-fetch provenance check found that **all 278 selected IDs had already failed quality in the original fetch manifests**. The excluded 10,550-object benchmark contains the quality-selected objects. This probe therefore examines a BTS remainder enriched for earlier quality failures; current public retrieval did not make these selected objects eligible. It does not show that public transient photometry or the wider BTS population is generally unusable. The earlier first-ten old-ID pilot is also not a generalizable estimate.

## Adapter control

The scientific leader requested a separate positive control after the zero eligibility count. The train-split object `ZTF17aaahrni` was queried after the candidate batch completed and was never added to the external cohort. Its stored benchmark events have 14 detections, 7 g and 7 r, and pass the original predicate. The current ALeRCE response contained 29 detections over its full returned history; applying the identical original first-100-day conversion again gave **14 detections, 7 g and 7 r, quality passed**. Independent counting directly from raw MJD and band fields matched the adapter. This control and selected-candidate schema/count spot checks found no adapter mismatch.

The original selection accepts `fid` 1/2/3 and applies no additional SNR floor. The same original converter, existing event validity check, and `flare.data.passes_quality` were used. A high full-history return count need not imply an adequate first window: for example, `ZTF18aaaokus` returned 1,446 detections but only one r-band point in its original first 100 days.

## Retrieval and reproducibility

The cohort used 275 network attempts: 274 HTTP 200 and one HTTP 404, plus three valid prior-cache reuses. No retries or HTTP 403/429 blocks occurred. Three workers, 20-second timeouts, and at most three attempts were configured. The positive control used one subsequent request at concurrency one. Every request/cache reuse is recorded in the raw append-only ledger, including timestamps and response hashes; unknown prior retrieval dates are explicitly distinguished from observed file mtimes.

A cache-only rerun succeeded with **zero new network requests** and preserved the hashes of 559 frozen-input, role-exclusion, eligible-ID, raw-response, and per-object record files. Nine synthetic boundary checks also passed, including exact 100-day maturity, i-band contribution to total counts, missing errors, minimum g/r counts, and absence of an added SNR floor. The executed cohort script SHA-256 was `a54149a434955e49d0fd312000effa0a99a172001399c57931eca3068892e236`.

```bash
cd /workspace/flare
export PYTHONPYCACHEPREFIX=/workspace/.cache/flare/pycache
export MPLCONFIGDIR=/workspace/.cache/flare/matplotlib
export XDG_CACHE_HOME=/workspace/.cache/flare
/workspace/.venvs/flare/bin/python scripts/uq_anytime/public_cohort.py --prepare-only
/workspace/.venvs/flare/bin/python scripts/uq_anytime/public_cohort.py
```

The script verifies frozen inputs and reuses completed object records on subsequent runs. Raw evidence lives outside Git at `/workspace/.research/flare-uq/external-stratified-v1`: frozen inputs, selected IDs, all benchmark and known fitting-role exclusions, photometry, request ledger, per-object records, and the frozen empty eligible-ID set.

Compact review evidence:

- [Summary](results/external_stratified_feasibility_summary.json)
- [Per-object eligibility and cadence CSV](results/external_stratified_feasibility_records.csv)
- [Source, exclusion, and model-role inventory](results/external_stratified_source_inventory.json)
- [Year and cadence diagnostics](results/external_stratified_cadence_diagnostics.json)
- [Retrieval audit](results/external_stratified_retrieval_audit.json)
- [Positive-control evidence](results/external_stratified_quality_control.json)
- [Execution and cache-rerun validation](results/external_stratified_execution_validation.json)

The selected ZTF identifier years span 2017–2025, with 99 from 2018 and 56 from 2019; the complete pool includes one 2026 ID. IAU designation-year distributions are also retained. These dates were diagnostics, not selection filters or verified physical event origins.

All 10,550 benchmark IDs and the union of training/early-stop IDs from all twenty research-model metadata files were excluded. The known fitting-role union contained 8,922 IDs and was wholly inside the benchmark. Original shipped/Fritz training identities remain unknown, so shipped-model independence is not established. Labels came from the frozen BTS spectral-type column using the existing six-class mapping, never ALeRCE classifier labels. This label-stratified feasibility probe is not population representative and establishes no external prediction performance, exchangeability, or prevalence estimate. The scientific leader received the exact failed threshold and control result for the next design decision.
