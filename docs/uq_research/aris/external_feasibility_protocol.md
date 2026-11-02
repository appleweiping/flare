# Frozen public external-cohort feasibility protocol

Frozen before any new photometry fetch for this probe: 2026-11-02T15:47:35.935619+00:00.
Scientific design: GPT-6-Astra leader; execution: GPT-6.1 Sol. Stage 1 measures public-data feasibility only. No models, prediction arrays, calibration data, model selection, or method tuning may be used.

## Candidate universe and exclusions

Use the retained current BTS sample explorer CSV at `/workspace/.research/flare-uq/external-feasibility/bts-current.csv` (SHA-256 `2a9a236b30c9c91be4450ff9a0c3c20a3b4b9c7ff7d4f5e843f2b9982dadd0ad`), without a fresh catalog refresh or a date-selected subset. BTS published spectral type is the class source; never substitute an ALeRCE classifier label. Map exact stripped type strings through the existing `scripts/make_splits.py:TYPE2CLS` six-class mapping. Unmapped, uncertain, missing, contradictory duplicate-ID classifications, novae, and out-of-scope types are excluded.

Exclude all 10,550 IDs in the union of `benchmark/splits/train.csv`, `val.csv`, `test.csv`, and `anomaly.csv`. Also exclude every `training_ids` and `early_stop_ids` entry in all twenty `study-v1/fold*/h*_metadata.json` records (five folds times four horizons). Read role identifiers only; do not open model or prediction files. Record separate exclusion flags and hashes/counts of the exact ID sets and metadata files. Current known own-model fit-ID union is 8,922, contained in the benchmark universe. Unknown original shipped/Fritz model training IDs are not supplied: this probe cannot establish independence from those shipped models.

Within each mapped class, sort by ascending SHA-256 of UTF-8 object ID, with ID as a secondary tie rule; inspect at most the first 64 available IDs per class, maximum 384. Freeze the selected IDs and label/source metadata before fetching, with class-specific pool counts and ID hashes. No replacing failed, immature, or low-quality IDs; no expanding the cap after seeing outcomes. Selection is stratified by known label and is not population representative. Retain ZTF identifier year and IAU designation year distributions for both pool and selected cohort; neither is a claim about the physical event origin.

## Frozen photometry and eligibility rules

Reference date is **2026-11-02 00:00:00 UTC**. Use public `https://api.alerce.online/ztf/v1/objects/<ID>/lightcurve` only. Match the original builder's detection selection exactly: `magpsf is not None` and `fid in (1,2,3)`. Convert with the original `scripts/build_bts_dataset.py:to_events`, whose clock starts at the first returned selected detection and retains `mjd <= first_mjd + 100`. This includes i-band detections where present. No extra SNR=3 floor, date cherry-pick, forced-photometry fallback, deduplication, or replacement of missing g/r is introduced.

Require a mature archive window: `first_returned_selected_mjd + 100 <= MJD(2026-11-02 00:00 UTC)`. A detection exactly on day 100 is not required. Record the returned first/last dates and the span; an old object ID alone does not establish maturity or the true event origin. Malformed/missing/nonfinite input or failure of the existing `flare.analysis.observed_events(events,100)` validation is a separate invalid-photometry disposition; do not silently drop bad rows to rescue eligibility.

Apply the exact existing `flare.data.passes_quality` predicate to the horizon-truncated events: at least **8 total detections**, at least **2 g**, at least **2 r**, at least **0 i**, and at least **2 observed bands**. These cuts match the original builder. Record raw returned count, retained detection counts at 5/10/20/100 days, per-band counts, distinct-epoch counts, gap diagnostics, and reason flags. Stage 1 does not extract model features or predict classes.

## Retrieval, persistence, and reruns

Use three concurrent requests (bounded within the authorized two-to-four range), 20-second timeout per attempt, at most three attempts for transport errors or HTTP 5xx, and bounded exponential backoff of 2 and 4 seconds. Record every attempt, status, timestamp, bytes, response SHA-256, elapsed time, and retry disposition. HTTP 404 is a final missing-source outcome. HTTP 403/429 stops new requests and records Retry-After; do not evade restrictions or hammer the service.

Reuse valid raw responses already cached under `external-feasibility/candidate_lightcurve_<ID>.json` or this run's own cache, recording source path, content hash, reuse timestamp, and original file mtime. Do not invent an exact earlier retrieval time where none was recorded. Keep raw network bytes, per-object records, frozen source IDs, and the append-only request ledger under `/workspace/.research/flare-uq/external-stratified-v1`; keep compact CSV/JSON evidence under `docs/uq_research/aris/results`. Existing matching run artifacts support a cache-only rerun; conflicting frozen inputs fail rather than overwrite.

## Decision rule and interpretation

After all fixed selections have a recorded outcome, report counts and exclusions without changing cuts. If **at least 100 total eligible** and **at least 10 eligible SLSN plus 10 eligible TDE**, report that the preflight threshold is met to the scientific leader for a separate decision. Otherwise report the exact limiting constraints and retain the artifacts. Pool availability before photometry is only 12 SLSN and 10 TDE outside the benchmark; those are not eligible counts. The previous lexicographic first-ten old-ID pilot with zero passing objects is not a generalizable feasibility estimate. Neither this feasibility probe nor passing its threshold establishes external model performance, shipped-model independence, exchangeability, or population prevalence.

## Frozen implementation references

- `scripts/make_splits.py`: `a28518e391c373b391e07ae2951edc875c15c0f6bca8b9b7a55aa528d42886ed`
- `scripts/build_bts_dataset.py`: `8e9d6d62eebc98f3d5a4d209c0e9afec764b49b47a5d524c09a73f5f9197620e`
- `flare/data.py`: `9de7d32227609899e420ccb00c6a05e25bc912a08cd7a624e164e018e510a7f7`
- `flare/fetch.py`: `a3bec3a106fc0323f51cfe20c34639206a7bf9510177f81766c407d42a2d8fd3`
- `flare/analysis.py`: `5183b36b0909d8e516ed1204414df66d3691830bfee896cdf90b4c756bab19d3`
