"""Run the frozen, exploratory rare-class lifetime-set/budget pilot without refits.

Inventory only (no policy outcomes):
  python scripts/uq_anytime/run_rare_class.py --study STUDY --inventory
Run only after the scientific protocol has been reviewed and frozen:
  python scripts/uq_anytime/run_rare_class.py --study STUDY --out NEW_DIRECTORY

The alpha grid is descriptive. There is no winner selection or certification on
these adaptively reused development objects. No production defaults are changed.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.uq_anytime.learned_policy import sha, validate_metadata, validate_study_complete, write_json
from scripts.uq_anytime.rare_class import (
    ALPHAS, ARMS, CLASSES, HORIZONS, _probabilities, _resources, _truth,
    calibrate_sets, first_singleton, nested_candidate_sets, summarize_sets,
)


def dataset_membership(study):
    """The source study's dataset labels, independently of hierarchy predictions."""
    with (Path(study) / "membership.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    membership = {}
    for row in rows:
        obj_id = row["obj_id"]
        if not obj_id or obj_id in membership or row["cls"] not in CLASSES:
            raise ValueError("invalid or duplicate dataset membership")
        fold = int(row["fold"])
        if fold not in range(5):
            raise ValueError("dataset membership fold must be between zero and four")
        membership[obj_id] = (fold, CLASSES.index(row["cls"]))
    return membership


def load_saved_fold(study, fold, membership):
    """Reuse validated source role/ID loading; exclude all anomaly rows.

    Truth arrays are cross-checked against source dataset membership and never
    replaced by hierarchy labels or probability argmax labels.
    """
    folder = Path(study) / f"fold{fold}"
    metadata = {h: json.loads((folder / f"h{h}_metadata.json").read_text()) for h in HORIZONS}
    with np.load(folder / "predictions.npz", allow_pickle=False) as saved:
        cal_ids, ev_ids = saved["calibration_ids"], saved["evaluation_ids"]
        roles = validate_metadata(metadata, cal_ids, ev_ids)
        nc, ne = len(cal_ids), len(ev_ids)
        stop = nc + ne
        ids = saved["obj_id"][:stop]
        if not np.array_equal(ids, np.concatenate([cal_ids, ev_ids])):
            raise ValueError("source prediction ID ordering mismatch")
        probs = saved["probabilities"][:stop].copy()
        nobs = saved["nobs"][:stop].copy()
        hierarchy_labels = saved["labels"][:stop].copy()
        calibration_y, evaluation_y = saved["calibration_y"], saved["evaluation_y"]
    probs = _probabilities(probs)
    if probs.shape != (stop, len(HORIZONS), len(CLASSES)):
        raise ValueError("source probability dimensions mismatch")
    _resources(nobs, HORIZONS, probs.shape[:2])
    _truth(calibration_y, nc, len(CLASSES))
    _truth(evaluation_y, ne, len(CLASSES))
    for role, object_ids, truth, expected_fold in (
            ("calibration", cal_ids, calibration_y, (fold + 1) % 5),
            ("evaluation", ev_ids, evaluation_y, fold)):
        if set(object_ids) != {obj_id for obj_id, (k, _) in membership.items() if k == expected_fold}:
            raise ValueError(f"{role} IDs differ from dataset fold membership")
        if any(membership[obj_id] != (expected_fold, int(y)) for obj_id, y in zip(object_ids, truth)):
            raise ValueError(f"{role} truth differs from dataset membership")
    model_hashes = {}
    for h, meta in metadata.items():
        for part in ("top", "branch"):
            digest = sha(folder / f"h{h}_{part}.txt")
            if digest != meta[f"{part}_sha256"]:
                raise ValueError("source model does not match saved metadata checksum")
            model_hashes[f"h{h}_{part}"] = digest
    inputs = dict(obj_id=ids, calibration_ids=cal_ids, evaluation_ids=ev_ids,
                  probabilities=probs, nobs=nobs, hierarchy_labels=hierarchy_labels,
                  calibration_y=calibration_y, evaluation_y=evaluation_y)
    provenance = dict(fold=fold, **roles, source_predictions_sha256=sha(folder / "predictions.npz"),
                      source_metadata_sha256={str(h): sha(folder / f"h{h}_metadata.json") for h in HORIZONS},
                      source_model_sha256=model_hashes)
    return inputs, provenance


def calibration_record(calibration):
    """Strict JSON uses null for +inf with an explicit infinity mask."""
    out = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
           for k, v in calibration.items() if k != "thresholds"}
    q = calibration["thresholds"]
    out["thresholds"] = [[float(x) if np.isfinite(x) else None for x in row] for row in q]
    out["threshold_positive_infinity"] = np.isposinf(q).tolist()
    return out


def _aggregate_groups(groups):
    """Pool counts/resource sums, never average per-fold fractions."""
    counts = ("n_objects", "n_selected", "n_unresolved", "n_wrong_automatic", "n_correct_automatic",
              "n_true_class_retained_unresolved", "n_true_class_lifetime_excluded",
              "n_true_class_consumed_prefix_excluded")
    out = {name: sum(g[name] for g in groups) for name in counts}
    n, selected = out["n_objects"], out["n_selected"]
    for target, numerator in (
        ("true_class_lifetime_exclusion", "n_true_class_lifetime_excluded"),
        ("true_class_consumed_prefix_exclusion", "n_true_class_consumed_prefix_excluded"),
        ("wrong_automatic_per_all", "n_wrong_automatic"),
        ("correct_automatic_per_all", "n_correct_automatic"),
        ("true_class_retained_unresolved_per_all", "n_true_class_retained_unresolved"),
        ("selection_coverage", "n_selected")):
        out[target] = out[numerator] / n if n else None
    out["selected_error"] = out["n_wrong_automatic"] / selected if selected else None
    unresolved = out["n_unresolved"]
    out["true_class_retained_given_unresolved"] = (out["n_true_class_retained_unresolved"] / unresolved
                                                 if unresolved else None)
    for metric, weight in (("mean_set_size_at_consumed", "n_objects"),
                           ("mean_set_size_unresolved", "n_unresolved")):
        denominator = out[weight]
        out[metric] = (sum(g[metric] * g[weight] for g in groups if g[weight]) / denominator
                       if denominator else None)
    out["mean_day_selected"] = (sum(g["mean_day_selected"] * g["n_selected"] for g in groups if g["n_selected"])
                                / selected if selected else None)
    out["mean_day_consumed_all"] = (sum(g["mean_day_consumed_all"] * g["n_objects"] for g in groups if g["n_objects"])
                                    / n if n else None)
    out["total_nobs_consumed"] = sum(g["total_nobs_consumed"] for g in groups)
    out["mean_nobs_consumed_all"] = out["total_nobs_consumed"] / n if n else None
    return out


def aggregate_folds(folds):
    combined = []
    keys = [(r["arm"], r["alpha"], r["budget_day"]) for r in folds[0]["records"]]
    indexed = [{(r["arm"], r["alpha"], r["budget_day"]): r for r in fold["records"]} for fold in folds]
    if any(set(index) != set(keys) or len(index) != len(keys) for index in indexed):
        raise ValueError("fold policy grids differ")
    for arm, alpha, budget_day in keys:
        metrics = [index[(arm, alpha, budget_day)]["metrics"] for index in indexed]
        combined.append(dict(arm=arm, alpha=alpha, budget_day=budget_day,
                             metrics=dict(overall=_aggregate_groups([m["overall"] for m in metrics]),
                                          by_true_class={c: _aggregate_groups([m["by_true_class"][c] for m in metrics])
                                                         for c in CLASSES})))
    return combined


def run_fold(inputs, out):
    """Compute all frozen arms and budgets; selection cannot access evaluation y."""
    out = Path(out)
    out.mkdir()
    nc = len(inputs["calibration_ids"])
    probs, nobs = inputs["probabilities"], inputs["nobs"]
    cal_y, ev_y = inputs["calibration_y"], inputs["evaluation_y"]
    np.savez_compressed(out / "inputs.npz", **inputs)
    paths, records, calibrations = {}, [], {}
    for name, method, mondrian in ARMS:
        for alpha_index, alpha in enumerate(ALPHAS):
            base_key = f"{name}_a{alpha_index}"
            if method != "fixedlook":
                cal = calibrate_sets(probs[:nc], cal_y, alpha, method=method, mondrian=mondrian)
                sets = nested_candidate_sets(probs[nc:], cal)
                paths[f"{base_key}_candidate_sets"] = sets
                paths[f"{base_key}_thresholds"] = cal["thresholds"]
                calibrations[base_key] = calibration_record(cal)
            for budget_index, budget in enumerate(HORIZONS):
                key = base_key
                if method == "fixedlook":
                    key = f"{base_key}_deadline{budget}"
                    cal = calibrate_sets(probs[:nc], cal_y, alpha, method=method, mondrian=mondrian,
                                         look_index=budget_index)
                    sets = nested_candidate_sets(probs[nc:], cal)
                    paths[f"{key}_candidate_sets"] = sets
                    paths[f"{key}_thresholds"] = cal["thresholds"]
                    calibrations[key] = calibration_record(cal)
                decision = first_singleton(sets, nobs[nc:], budget_day=budget)
                for field, values in decision.items():
                    paths[f"{key}_b{budget}_{field}"] = values
                records.append(dict(arm=name, alpha=alpha, alpha_index=alpha_index, budget_day=budget,
                                    calibration_key=key, metrics=summarize_sets(sets, decision, ev_y)))
    np.savez_compressed(out / "sets_decisions.npz", evaluation_ids=inputs["evaluation_ids"], **paths)
    write_json(out / "calibration.json", calibrations)
    return dict(n_calibration=nc, n_evaluation=len(ev_y), records=records)


def inventory(inputs_by_fold):
    return dict(horizons=HORIZONS, classes=CLASSES, folds=[dict(
        fold=k, n_calibration=len(data["calibration_ids"]), n_evaluation=len(data["evaluation_ids"]),
        probabilities_shape=list(data["probabilities"].shape),
        n_calibration_by_class=np.bincount(data["calibration_y"], minlength=len(CLASSES)).tolist(),
        n_evaluation_by_class=np.bincount(data["evaluation_y"], minlength=len(CLASSES)).tolist())
        for k, (data, _) in enumerate(inputs_by_fold)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--inventory", action="store_true")
    args = parser.parse_args()
    if not args.inventory and args.out is None:
        parser.error("--out is required unless --inventory is used")
    # Never overwrite an existing pilot or source artifact directory.
    if not args.inventory and args.out.exists():
        raise ValueError("output directory already exists; choose a new directory")
    source_result_sha = validate_study_complete(args.study)
    membership = dataset_membership(args.study)
    inputs_by_fold = [load_saved_fold(args.study, fold, membership) for fold in range(5)]
    seen = [obj_id for data, _ in inputs_by_fold for obj_id in data["evaluation_ids"]]
    if len(seen) != len(set(seen)) or set(seen) != set(membership):
        raise ValueError("evaluation objects must cover dataset membership exactly once")
    source_inventory = inventory(inputs_by_fold)
    if args.inventory:
        print(json.dumps(source_inventory, indent=2, allow_nan=False))
        return
    args.out.mkdir(parents=True)
    code_files = [Path(__file__), Path(__file__).with_name("rare_class.py"),
                  Path(__file__).with_name("learned_policy.py"), Path(__file__).with_name("policy.py")]
    code_dir = args.out / "code"
    code_dir.mkdir()
    for path in code_files:
        (code_dir / path.name).write_bytes(path.read_bytes())
    provenance = dict(
        interpretation="Posthoc exploratory pilot on reused development objects; no confirmatory claim",
        source_study=str(args.study.resolve()), source_results_sha256=source_result_sha,
        source_membership_sha256=sha(args.study / "membership.csv"),
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        code_sha256={path.name: sha(path) for path in code_files},
        package_versions={p: importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-learn")},
        horizons=HORIZONS, classes=CLASSES, alphas=ALPHAS, arms=[a[0] for a in ARMS],
        budget_days=HORIZONS, winner_selection=False, classifier_retrained=False,
        test_outcomes_read=False, anomaly_rows_used=False, truth_source="dataset membership.csv, checked against saved y",
        score="1-probability_of_candidate_class", finite_sample_quantile="ceil((n+1)*(1-alpha)); inf if k>n",
        empty_set_policy="unresolved; continue to budget and abstain", decision="retain first singleton",
        lifetime_exclusion_scope="full scheduled set path, diagnostic even after decision; fixedlook set held after deadline",
        fixedlook_guarantee="full alpha separately at each prespecified budget; no adaptive any-time full-alpha guarantee",
        fixedlook_cost="all objects consume exactly the prespecified deadline",
        consumed_prefix_exclusion_scope="only looks through consumed stop index",
        observation_cost="cumulative nobs at consumed stop; not sum of cumulative nobs across looks",
        stop_index="consumed look for every object; decision_index=-1 for abstainers",
        source_inventory=source_inventory)
    write_json(args.out / "provenance.json", provenance)
    folds = []
    for fold, (inputs, fold_provenance) in enumerate(inputs_by_fold):
        folder = args.out / f"fold{fold}"
        result = dict(fold=fold, **run_fold(inputs, folder))
        fold_provenance["artifacts_sha256"] = {p.name: sha(p) for p in folder.iterdir() if p.is_file()}
        write_json(folder / "provenance.json", fold_provenance)
        write_json(folder / "result.json", result)
        folds.append(result)
        print(f"fold={fold} saved {len(result['records'])} descriptive arm/alpha/budget records", flush=True)
    artifact_hashes = {str(path.relative_to(args.out)): sha(path)
                       for path in sorted(args.out.rglob("*")) if path.is_file()}
    write_json(args.out / "results.json", dict(provenance=provenance, folds=folds, pooled=aggregate_folds(folds),
                                               artifacts_sha256=artifact_hashes))
    (args.out / "COMPLETE").write_text(sha(args.out / "results.json") + "\n")


if __name__ == "__main__":
    main()
