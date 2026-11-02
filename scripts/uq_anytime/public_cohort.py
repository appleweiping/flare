"""Frozen BTS/ALeRCE external feasibility only; never loads models or predictions."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import re
import sys
from threading import Event, Lock
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from flare.analysis import observed_events
from flare.data import COL, passes_quality

CLASSES = ("SN_Ia", "SN_CC", "SLSN", "AGN", "TDE", "CV")
BASE = Path("/workspace/.research/flare-uq")
OUT = BASE / "external-stratified-v1"
PROTOCOL = REPO / "docs/uq_research/aris/external_feasibility_protocol.md"
PROTOCOL_SHA = "d2efff7f134ffa2577b3eeed41e43682e1d4dd8b5137da89bbbf654664328cd2"
EVIDENCE = REPO / "docs/uq_research/aris/results"
REFERENCE_UTC = datetime(2026, 11, 2, tzinfo=timezone.utc)
REFERENCE_MJD = (REFERENCE_UTC - datetime(1858, 11, 17, tzinfo=timezone.utc)).total_seconds() / 86400


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def id_sha(ids):
    return hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode()).hexdigest()


def load_script(name, relative):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MAPPING = load_script("bts_mapping", "scripts/make_splits.py").TYPE2CLS
BUILDER = load_script("bts_original_builder", "scripts/build_bts_dataset.py")


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def freeze_or_verify(path, data):
    if path.exists():
        if json.loads(path.read_text()) != data:
            raise ValueError(f"frozen input differs; preserving {path}")
    else:
        with path.open("x") as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write("\n")


def years(rows, field):
    return dict(sorted(Counter(row[field] for row in rows).items()))


def prepare():
    if sha(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("protocol changed after freeze")
    OUT.mkdir(parents=True, exist_ok=True)
    benchmark = set()
    provenance = {}
    for split in ("train", "val", "test", "anomaly"):
        path = REPO / f"benchmark/splits/{split}.csv"
        benchmark.update(pd.read_csv(path, usecols=["obj_id"]).obj_id.astype(str))
        provenance[str(path.relative_to(REPO))] = sha(path)
    if len(benchmark) != 10550:
        raise ValueError("benchmark universe differs from frozen 10,550 IDs")
    fitting = set()
    role_counts = []
    for fold in range(5):
        for horizon in (5, 10, 20, 100):
            path = BASE / f"study-v1/fold{fold}/h{horizon}_metadata.json"
            metadata = json.loads(path.read_text())
            training, early = metadata["training_ids"], metadata["early_stop_ids"]
            if not isinstance(training, list) or not isinstance(early, list):
                raise ValueError(f"missing fitting-role IDs: {path}")
            fitting.update(training)
            fitting.update(early)
            role_counts.append(dict(path=str(path), sha256=sha(path), training=len(training), early_stop=len(early)))
    catalog = BASE / "external-feasibility/bts-current.csv"
    rows = pd.read_csv(io.StringIO(catalog.read_text().lstrip()), dtype=str, keep_default_na=False).to_dict("records")
    grouped = {}
    for row in rows:
        oid = row["ZTFID"].strip()
        grouped.setdefault(oid, []).append(row)
    pool = []
    exclusion_counts = Counter()
    for oid, candidates in sorted(grouped.items()):
        kinds = {row["type"].strip() for row in candidates}
        if oid in benchmark:
            exclusion_counts["benchmark_id"] += 1
            continue
        if oid in fitting:
            exclusion_counts["known_own_model_fit_id"] += 1
            continue
        if not re.fullmatch(r"ZTF\d{2}[a-z]{7}", oid):
            exclusion_counts["malformed_id"] += 1
            continue
        if len(kinds) != 1:
            exclusion_counts["contradictory_duplicate_label"] += 1
            continue
        kind = next(iter(kinds))
        if kind not in MAPPING:
            exclusion_counts[f"unmapped_type:{kind}"] += 1
            continue
        row = min(candidates, key=lambda r: json.dumps(r, sort_keys=True))
        iau = re.search(r"(?:AT|SN)(20\d{2})", row["IAUID"])
        pool.append(dict(obj_id=oid, cls=MAPPING[kind], bts_type=kind,
                         iau_id=row["IAUID"], ztf_id_year="20" + oid[3:5],
                         iau_designation_year=iau.group(1) if iau else "unknown",
                         id_sha256=hashlib.sha256(oid.encode()).hexdigest(),
                         source_peak_time=row["peakt"], source_peak_mag=row["peakmag"]))
    selected = []
    for cls in CLASSES:
        selected.extend(sorted((row for row in pool if row["cls"] == cls),
                               key=lambda row: (row["id_sha256"], row["obj_id"]))[:64])
    sources = {relative: sha(REPO / relative) for relative in (
        "scripts/make_splits.py", "scripts/build_bts_dataset.py", "flare/data.py",
        "flare/fetch.py", "flare/analysis.py")}
    freeze = dict(protocol_sha256=PROTOCOL_SHA, reference_utc=REFERENCE_UTC.isoformat(),
                  catalog_sha256=sha(catalog), candidate_inventory_sha256=sha(BASE / "external-feasibility/labelled_candidate_ids.txt"),
                  benchmark_ids=len(benchmark), benchmark_ids_sha256=id_sha(benchmark),
                  known_own_model_fit_ids=len(fitting), fitting_ids_sha256=id_sha(fitting),
                  fitting_ids_outside_benchmark=len(fitting - benchmark), fitting_metadata=role_counts,
                  benchmark_input_sha256=provenance, implementation_reference_sha256=sources,
                  source_rows=len(rows), distinct_source_ids=len(grouped), exclusion_counts=dict(exclusion_counts),
                  pool_counts=dict(Counter(row["cls"] for row in pool)),
                  selected_counts=dict(Counter(row["cls"] for row in selected)),
                  selected_ids_sha256=id_sha(row["obj_id"] for row in selected),
                  pool_years={key: years(pool, key) for key in ("ztf_id_year", "iau_designation_year")},
                  selected_years={key: years(selected, key) for key in ("ztf_id_year", "iau_designation_year")},
                  selected=selected, shipped_model_fit_ids_known=False,
                  model_predictions_loaded=False, calibration_data_loaded=False)
    freeze_or_verify(OUT / "frozen_inputs.json", freeze)
    freeze_or_verify(OUT / "role_exclusions.json", dict(benchmark_ids=sorted(benchmark), known_own_model_fit_ids=sorted(fitting)))
    pd.DataFrame(selected).to_csv(OUT / "selected_ids.csv", index=False)
    marker = OUT / "selection_frozen_at.json"
    if not marker.exists():
        freeze_or_verify(marker, dict(frozen_at_utc=now(), protocol_sha256=PROTOCOL_SHA,
                                     selected_ids_sha256=freeze["selected_ids_sha256"]))
    print(json.dumps(dict(prepared=True, selected=len(selected), per_class=freeze["selected_counts"], utc=now())), flush=True)
    return freeze


def finite_number(value):
    return value is not None and math.isfinite(float(value))


def assess(payload):
    raw = payload["detections"]
    if not isinstance(raw, list):
        raise ValueError("detections is not a list")
    # This is the original dataset builder's selection, not the live SNR floor.
    chosen = [row for row in raw if row.get("magpsf") is not None and row.get("fid") in (1, 2, 3)]
    base = dict(returned_detections=len(raw), original_selected_detections=len(chosen))
    if not chosen:
        return dict(base, eligible=False, disposition="no_selected_detections", mature=False)
    required = ("mjd", "magpsf", "sigmapsf")
    if any(not all(finite_number(row.get(key)) for key in required) for row in chosen):
        return dict(base, eligible=False, disposition="invalid_photometry", mature=False)
    tuples = [(float(row["mjd"]), row["fid"], float(row["magpsf"]), float(row["sigmapsf"])) for row in chosen]
    first = min(row[0] for row in tuples)
    last = max(row[0] for row in tuples)
    mature = bool(first + 100 <= REFERENCE_MJD)
    events = BUILDER.to_events(tuples)
    try:
        events = observed_events(events, 100)
    except ValueError as error:
        return dict(base, eligible=False, disposition="invalid_photometry", mature=mature,
                    first_returned_mjd=first, last_returned_mjd=last, validation_error=str(error))
    horizons = {}
    for horizon in (5, 10, 20, 100):
        prefix = events[events[:, COL["dt"]] <= horizon]
        bands = prefix[:, COL["band_id"]]
        times = np.unique(prefix[:, COL["dt"]].astype(float))
        gaps = np.diff(times)
        horizons[str(horizon)] = dict(n=len(prefix), g=int((bands == 0).sum()),
                                     r=int((bands == 1).sum()), i=int((bands == 2).sum()),
                                     distinct_epochs=len(times), span=float(times[-1] - times[0]) if len(times) else 0,
                                     median_gap=float(np.median(gaps)) if len(gaps) else None,
                                     max_gap=float(gaps.max()) if len(gaps) else None,
                                     passes_quality=bool(passes_quality(prefix)))
    quality = horizons["100"]["passes_quality"]
    failures = []
    if not mature:
        failures.append("immature_100d_window")
    if horizons["100"]["n"] < 8:
        failures.append("fewer_than_8_total")
    if horizons["100"]["g"] < 2:
        failures.append("fewer_than_2_g")
    if horizons["100"]["r"] < 2:
        failures.append("fewer_than_2_r")
    return dict(base, eligible=bool(mature and quality), mature=mature,
                passes_original_quality=quality, disposition="eligible" if mature and quality else "ineligible",
                failure_reasons=failures, first_returned_mjd=first, last_returned_mjd=last,
                returned_span_days=last - first, horizons=horizons)


def fetch_all(freeze):
    for folder in ("photometry", "retrieval", "records"):
        (OUT / folder).mkdir(exist_ok=True)
    lock, stopped = Lock(), Event()

    def ledger(record):
        with lock:
            with (OUT / "request_ledger.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")

    def one(row):
        oid = row["obj_id"]
        target = OUT / "photometry" / f"{oid}.json"
        retrieval_path = OUT / "retrieval" / f"{oid}.json"
        record_path = OUT / "records" / f"{oid}.json"
        url = f"https://api.alerce.online/ztf/v1/objects/{oid}/lightcurve"
        result = dict(row, url=url, protocol_sha256=PROTOCOL_SHA)
        if record_path.exists():
            previous = json.loads(record_path.read_text())
            if previous.get("protocol_sha256") != PROTOCOL_SHA or any(previous.get(key) != value for key, value in row.items()):
                raise ValueError(f"existing object record conflicts with frozen selection: {oid}")
            if "response_sha256" in previous and (not target.exists() or sha(target) != previous["response_sha256"]):
                raise ValueError(f"cached bytes changed since frozen object assessment: {oid}")
            ledger(dict(obj_id=oid, event="frozen_record_reuse", observed_utc=now(), record_sha256=sha(record_path)))
            return previous
        payload = None
        prior = BASE / "external-feasibility" / f"candidate_lightcurve_{oid}.json"
        for cache in (target, prior):
            if not cache.exists():
                continue
            try:
                body = cache.read_bytes()
                candidate = json.loads(body)
                if not isinstance(candidate, dict) or not isinstance(candidate.get("detections"), list):
                    raise ValueError("cached response lacks a detections list")
                payload = candidate
                event = dict(obj_id=oid, url=url, event="cache_reuse", observed_utc=now(),
                             source_cache_path=str(cache), source_file_mtime_utc=datetime.fromtimestamp(cache.stat().st_mtime, timezone.utc).isoformat(),
                             bytes=len(body), response_sha256=hashlib.sha256(body).hexdigest(),
                             original_retrieval_time_known=retrieval_path.exists())
                ledger(event)
                if not target.exists():
                    with target.open("xb") as stream:
                        stream.write(body)
                if not retrieval_path.exists():
                    write_json(retrieval_path, event)
                result.update(response_sha256=event["response_sha256"], cache_reused=True)
                break
            except (ValueError, OSError) as error:
                ledger(dict(obj_id=oid, event="invalid_cache", source_cache_path=str(cache), observed_utc=now(), error=str(error)))
                if cache == target:
                    raise ValueError(f"run cache invalid; preserve and investigate {cache}") from error
        if payload is None:
            for attempt in range(1, 4):
                if stopped.is_set():
                    result.update(eligible=False, disposition="not_requested_after_rate_or_access_block")
                    break
                event = dict(obj_id=oid, url=url, event="request_attempt", attempt=attempt, started_utc=now())
                start = time.monotonic()
                retry = False
                try:
                    request = Request(url, headers={"User-Agent": "FLARE-public-feasibility/1.0", "Accept": "application/json"})
                    with urlopen(request, timeout=20) as response:
                        body = response.read()
                        event.update(status=response.status, bytes=len(body), response_sha256=hashlib.sha256(body).hexdigest())
                    candidate = json.loads(body)
                    if not isinstance(candidate, dict) or not isinstance(candidate.get("detections"), list):
                        raise ValueError("response lacks a detections list")
                    if target.exists() and target.read_bytes() != body:
                        raise ValueError("different response already exists; preserving cache")
                    if not target.exists():
                        with target.open("xb") as stream:
                            stream.write(body)
                    event["retrieved_utc"] = now()
                    write_json(retrieval_path, event)
                    result.update(response_sha256=event["response_sha256"], cache_reused=False)
                    payload = candidate
                except HTTPError as error:
                    event.update(status=error.code, error=str(error), retry_after=error.headers.get("Retry-After"))
                    if error.code in (403, 429):
                        stopped.set()
                    retry = error.code >= 500 and attempt < 3
                    result.update(eligible=False, disposition=f"http_{error.code}")
                except (OSError, ValueError) as error:
                    event.update(status="error", error=f"{type(error).__name__}: {error}")
                    retry = isinstance(error, OSError) and attempt < 3
                    result.update(eligible=False, disposition="retrieval_error")
                finally:
                    event.update(elapsed_seconds=round(time.monotonic() - start, 3), finished_utc=now(), will_retry=retry)
                    ledger(event)
                if payload is not None or not retry:
                    break
                time.sleep(2 ** attempt)
        if payload is not None:
            try:
                result.update(assess(payload))
            except (ValueError, TypeError, KeyError, OverflowError) as error:
                result.update(eligible=False, disposition="invalid_photometry", validation_error=f"{type(error).__name__}: {error}")
        result["assessed_utc"] = now()
        write_json(record_path, result)
        return result

    records = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(one, row) for row in freeze["selected"]]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            if len(records) % 12 == 0 or len(records) == len(futures):
                print(json.dumps(dict(utc=now(), completed=len(records), total=len(futures),
                                      eligible=sum(record["eligible"] for record in records),
                                      rate_or_access_block=stopped.is_set())), flush=True)
    return records


def summarize(freeze, records):
    order = {row["obj_id"]: i for i, row in enumerate(freeze["selected"])}
    records.sort(key=lambda row: order[row["obj_id"]])
    eligible = [row for row in records if row["eligible"]]
    counts = {cls: dict(pool=freeze["pool_counts"].get(cls, 0), selected=freeze["selected_counts"].get(cls, 0),
                        eligible=sum(row["cls"] == cls for row in eligible),
                        dispositions=dict(Counter(row["disposition"] for row in records if row["cls"] == cls))) for cls in CLASSES}
    threshold = len(eligible) >= 100 and counts["SLSN"]["eligible"] >= 10 and counts["TDE"]["eligible"] >= 10
    summary = dict(completed_utc=now(), protocol_sha256=PROTOCOL_SHA, script_sha256=sha(__file__),
                   selected=len(records), eligible=len(eligible), per_class=counts,
                   threshold_met=threshold, threshold=dict(total=100, SLSN=10, TDE=10),
                   dispositions=dict(Counter(row["disposition"] for row in records)),
                   failure_reasons=dict(Counter(reason for row in records for reason in row.get("failure_reasons", []))),
                   eligible_ids_sha256=id_sha(row["obj_id"] for row in eligible),
                   eligible_years={key: years(eligible, key) for key in ("ztf_id_year", "iau_designation_year")},
                   cache_reused=sum(row.get("cache_reused", False) for row in records),
                   model_predictions_loaded=False, calibration_data_loaded=False,
                   interpretation="Label-stratified feasibility probe; not population representative; original shipped-model training independence unknown")
    freeze_or_verify(OUT / "eligible_ids_frozen.json", dict(protocol_sha256=PROTOCOL_SHA,
                                                           eligible_ids=sorted(row["obj_id"] for row in eligible)))
    write_json(OUT / "summary.json", summary)
    write_json(OUT / "all_records.json", records)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    write_json(EVIDENCE / "external_stratified_feasibility_summary.json", summary)
    compact = []
    for row in records:
        item = {key: row.get(key) for key in ("obj_id", "cls", "bts_type", "iau_id", "ztf_id_year", "iau_designation_year", "eligible", "disposition", "mature", "first_returned_mjd", "last_returned_mjd", "returned_span_days", "cache_reused", "response_sha256")}
        item["failure_reasons"] = ";".join(row.get("failure_reasons", []))
        for horizon, diagnostics in row.get("horizons", {}).items():
            item.update({f"h{horizon}_{key}": value for key, value in diagnostics.items()})
        compact.append(item)
    pd.DataFrame(compact).to_csv(EVIDENCE / "external_stratified_feasibility_records.csv", index=False)
    write_json(EVIDENCE / "external_stratified_source_inventory.json", {key: value for key, value in freeze.items() if key != "selected"})
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    options = parser.parse_args()
    freeze = prepare()
    if not options.prepare_only:
        summarize(freeze, fetch_all(freeze))


if __name__ == "__main__":
    main()
