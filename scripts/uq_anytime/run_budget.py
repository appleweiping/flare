"""Frozen exploratory causal checkpoint-budget study; no classifier refitting.

Run only after code/protocol review:
  python scripts/uq_anytime/run_budget.py --study STUDY --learned LEARNED --out NEWDIR
Continuation regressions use proper-training early-stop objects exclusively.
Evaluation truth enters descriptive metrics only, never scheduler/scorer inputs.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.uq_anytime.budget import allocate_query_budget
from scripts.uq_anytime.learned_budget import (
    VARIANTS, continuation_scorer, fit_continuation_models, model_record, training_targets,
)
from scripts.uq_anytime.learned_policy import sha, validate_study_complete, write_json
from scripts.uq_anytime.rare_class import (
    CLASSES, HORIZONS, _probabilities, _resources, _truth, calibrate_sets, nested_candidate_sets, summarize_sets,
)
from scripts.uq_anytime.run_rare_class import (
    _aggregate_groups, calibration_record, dataset_membership, load_saved_fold,
)

ALPHAS = (.1, .2)
BUDGET_MULTIPLIERS = (0, .25, .5, 1, 2, 3)
HEURISTICS = ("round_robin", "entropy", "candidate_count", "rare_mass", "confidence")
RANDOM_SEEDS = (0, 1, 2)


def load_gate_training(learned, fold, roles, membership):
    folder = Path(learned) / f"fold{fold}"
    source = folder / "gate_training.npz"
    provenance = json.loads((folder / "provenance.json").read_text())
    digest = sha(source)
    if digest != provenance["gate_training_sha256"]:
        raise ValueError("source gate training archive checksum mismatch")
    with np.load(source, allow_pickle=False) as saved:
        gate = {key: saved[key].copy() for key in ("obj_id", "probabilities", "labels", "nobs", "y")}
    ids = gate["obj_id"]
    if not np.array_equal(ids, roles["gate_training_ids"]):
        raise ValueError("gate training IDs/order differ from saved classifier early-stop membership")
    if len(set(ids)) != len(ids) or set(ids) & (set(roles["calibration_ids"]) | set(roles["evaluation_ids"])):
        raise ValueError("gate training roles contaminated")
    _probabilities(gate["probabilities"])
    if gate["probabilities"].shape != (len(ids), 4, 6):
        raise ValueError("gate training dimensions mismatch")
    _resources(gate["nobs"], HORIZONS, (len(ids), 4))
    _truth(gate["y"], len(ids), 6)
    if any(obj_id not in membership or membership[obj_id][1] != int(y) for obj_id, y in zip(ids, gate["y"])):
        raise ValueError("gate training truth differs from source dataset membership")
    return gate, dict(source_gate_training_sha256=digest,
                      source_gate_provenance_sha256=sha(folder / "provenance.json"), continuation_training_ids=ids.tolist())


def prediction_diagnostics(predictions, y, selected):
    """Emitted-label precision is distinct from metrics grouped by TRUE class."""
    predictions, selected = np.asarray(predictions), np.asarray(selected)
    y = _truth(y, len(predictions), 6)
    if predictions.shape != y.shape or selected.shape != y.shape or selected.dtype != np.bool_:
        raise ValueError("invalid prediction diagnostics dimensions")
    if ((predictions[~selected] != -1).any() or (predictions[selected] < 0).any()
            or (predictions[selected] >= 6).any()):
        raise ValueError("invalid automatic predictions")
    confusion = np.zeros((6, 6), dtype=int)
    np.add.at(confusion, (y[selected], predictions[selected]), 1)
    groups = {}
    for cls, name in enumerate(CLASSES):
        emitted = int(confusion[:, cls].sum())
        correct = int(confusion[cls, cls])
        groups[name] = dict(n_emitted=emitted, n_correct=correct, n_wrong=emitted-correct,
                            precision=correct/emitted if emitted else None,
                            error_given_emitted=(emitted-correct)/emitted if emitted else None)
    return dict(confusion_true_rows_predicted_columns=confusion.tolist(), by_predicted_class=groups)


def _add_derived_metrics(metrics):
    groups = list(metrics["by_true_class"].values())
    represented = [g for g in groups if g["n_objects"]]
    metrics["macro_true_class_correct_automatic_rate"] = (
        float(np.mean([g["correct_automatic_per_all"] for g in represented])) if represented else None)
    metrics["macro_true_class_wrong_automatic_rate"] = (
        float(np.mean([g["wrong_automatic_per_all"] for g in represented])) if represented else None)
    metrics["n_classes_represented_for_macro"] = len(represented)
    return metrics


def summarize_allocation(sets, nobs, allocation, y, allocated_tokens):
    selected = allocation.status == "automatic"
    rows = np.arange(len(selected))
    final = allocation.final_look
    decision = dict(selected=selected, prediction=allocation.predictions, stop_index=final,
                    consumed_day=np.asarray(HORIZONS)[final], nobs_consumed=nobs[rows, final])
    metrics = _add_derived_metrics(summarize_sets(sets, decision, y))
    metrics.update(prediction_diagnostics(allocation.predictions, y, selected))
    metrics["resources"] = dict(initial_checkpoint_queries=len(selected), allocated_extra_tokens=int(allocated_tokens),
                                consumed_extra_tokens=allocation.consumed_tokens,
                                unused_extra_tokens=int(allocated_tokens)-allocation.consumed_tokens,
                                total_checkpoint_queries=len(selected)+allocation.consumed_tokens)
    metrics["status_counts"] = {status: int(np.sum(allocation.status == status))
                                for status in ("automatic", "empty", "exhausted", "budget")}
    return metrics


def save_allocation(path, allocation, sets, nobs, evaluation_ids):
    final, rows = allocation.final_look, np.arange(len(evaluation_ids))
    actions = allocation.actions
    arrays = dict(evaluation_ids=evaluation_ids, final_look=final, prediction=allocation.predictions,
                  status=allocation.status, consumed_day=np.asarray(HORIZONS)[final],
                  nobs_consumed=nobs[rows, final], set_size_at_stop=sets[rows, final].sum(axis=1),
                  final_candidate_sets=sets[rows, final])
    for field in ("token", "index", "from_look", "to_look"):
        arrays[f"action_{field}"] = np.array([getattr(a, field) for a in actions], dtype=int)
    arrays["action_obj_id"] = np.array([a.obj_id for a in actions], dtype=str)
    arrays["action_observed_nobs"] = np.array([a.observed_nobs for a in actions], dtype=float)
    arrays["action_score"] = np.array([np.nan if a.score is None else a.score for a in actions], dtype=float)
    np.savez_compressed(path, **arrays)


def run_budget_fold(inputs, gate, out):
    """All frozen cells; only the final summary call receives evaluation truth."""
    out = Path(out)
    out.mkdir()
    decision_dir = out / "decisions"
    decision_dir.mkdir()
    np.savez_compressed(out / "inputs.npz", **inputs)
    np.savez_compressed(out / "gate_training.npz", **gate)
    nc, ne = len(inputs["calibration_ids"]), len(inputs["evaluation_ids"])
    cal_probs, evaluation_probs = inputs["probabilities"][:nc], inputs["probabilities"][nc:]
    nobs = inputs["nobs"][nc:]
    all_models, metadata, records, path_arrays = {}, {}, [], {}
    for regime, mondrian in (("pooled", False), ("mondrian", True)):
        for alpha_index, alpha in enumerate(ALPHAS):
            key = f"envelope_{regime}_a{alpha_index}"
            cal = calibrate_sets(cal_probs, inputs["calibration_y"], alpha, mondrian=mondrian)
            sets = nested_candidate_sets(evaluation_probs, cal)
            gate_sets = nested_candidate_sets(gate["probabilities"], cal)
            targets = training_targets(gate_sets, gate["y"])
            np.savez_compressed(out / f"{key}_training_targets.npz", **targets)
            models = fit_continuation_models(gate["probabilities"], gate["nobs"], gate["labels"], gate["y"], gate_sets)
            all_models[key] = models
            metadata[key] = dict(calibration=calibration_record(cal),
                                 models={name: model_record(model) for name, model in models.items()})
            path_arrays[f"{key}_evaluation_candidate_sets"] = sets
            path_arrays[f"{key}_training_candidate_sets"] = gate_sets
            path_arrays[f"{key}_thresholds"] = cal["thresholds"]
            # Reuse deterministic callbacks across budgets, memoizing ONLY
            # prefixes actually exposed in previous callback invocations.
            policies = [(name, name, None, None) for name in HEURISTICS]
            policies += [("random", f"random_seed{seed}", seed, None) for seed in RANDOM_SEEDS]
            policies += [("custom", name, None, continuation_scorer(models[name])) for name in VARIANTS]
            for multiplier in BUDGET_MULTIPLIERS:
                budget = int(np.floor(multiplier * ne))
                for policy, policy_name, seed, scorer in policies:
                    allocation = allocate_query_budget(
                        sets, evaluation_probs, nobs, inputs["evaluation_ids"], budget,
                        policy=policy, seed=0 if seed is None else seed, scorer=scorer)
                    filename = f"{key}_b{multiplier:g}_{policy_name}.npz"
                    path = decision_dir / filename
                    save_allocation(path, allocation, sets, nobs, inputs["evaluation_ids"])
                    metrics = summarize_allocation(sets, nobs, allocation, inputs["evaluation_y"], budget)
                    records.append(dict(regime=regime, alpha=alpha, policy=policy_name, random_seed=seed,
                                        budget_multiplier=multiplier, allocated_extra_tokens=budget,
                                        decision_artifact=f"decisions/{filename}", decision_sha256=sha(path), metrics=metrics))
    with (out / "models.pkl").open("wb") as stream:
        pickle.dump(all_models, stream, protocol=pickle.HIGHEST_PROTOCOL)
    write_json(out / "model_metadata.json", metadata)
    np.savez_compressed(out / "candidate_paths.npz", **path_arrays)
    return dict(n_gate_training=len(gate["obj_id"]), n_calibration=nc, n_evaluation=ne, records=records)


def aggregate_budget_folds(folds):
    keys = [(r["regime"], r["alpha"], r["policy"], r["budget_multiplier"]) for r in folds[0]["records"]]
    indexed = [{(r["regime"], r["alpha"], r["policy"], r["budget_multiplier"]): r for r in fold["records"]}
               for fold in folds]
    if any(set(index) != set(keys) or len(index) != len(keys) for index in indexed):
        raise ValueError("fold allocation grids differ")
    pooled = []
    for regime, alpha, policy, multiplier in keys:
        metrics = [index[(regime, alpha, policy, multiplier)]["metrics"] for index in indexed]
        aggregate = dict(overall=_aggregate_groups([m["overall"] for m in metrics]),
                         by_true_class={c: _aggregate_groups([m["by_true_class"][c] for m in metrics]) for c in CLASSES})
        _add_derived_metrics(aggregate)
        confusion = np.sum([m["confusion_true_rows_predicted_columns"] for m in metrics], axis=0)
        aggregate["confusion_true_rows_predicted_columns"] = confusion.tolist()
        aggregate["by_predicted_class"] = {}
        for c, cls in enumerate(CLASSES):
            emitted, correct = int(confusion[:, c].sum()), int(confusion[c, c])
            aggregate["by_predicted_class"][cls] = dict(n_emitted=emitted, n_correct=correct, n_wrong=emitted-correct,
                                                       precision=correct/emitted if emitted else None,
                                                       error_given_emitted=(emitted-correct)/emitted if emitted else None)
        aggregate["resources"] = {k: sum(m["resources"][k] for m in metrics) for k in metrics[0]["resources"]}
        aggregate["status_counts"] = {k: sum(m["status_counts"][k] for m in metrics) for k in metrics[0]["status_counts"]}
        pooled.append(dict(regime=regime, alpha=alpha, policy=policy, budget_multiplier=multiplier, metrics=aggregate))
    return pooled


def write_compact(out, pooled):
    overall, classwise, emitted = [], [], []
    for record in pooled:
        context = {key: record[key] for key in ("regime", "alpha", "policy", "budget_multiplier")}
        metrics = record["metrics"]
        overall.append(dict(**context, **metrics["overall"], **metrics["resources"],
                            macro_true_class_correct_automatic_rate=metrics["macro_true_class_correct_automatic_rate"],
                            macro_true_class_wrong_automatic_rate=metrics["macro_true_class_wrong_automatic_rate"]))
        for cls in CLASSES:
            classwise.append(dict(**context, true_class=cls, **metrics["by_true_class"][cls]))
            emitted.append(dict(**context, predicted_class=cls, **metrics["by_predicted_class"][cls]))
    for name, rows in (("compact_frontier.csv", overall), ("classwise_frontier.csv", classwise), ("emitted_label_precision.csv", emitted)):
        with (Path(out) / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--learned", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("output directory already exists; choose a new directory")
    source_hash = validate_study_complete(args.study)
    learned_hash = validate_study_complete(args.learned)
    membership = dataset_membership(args.study)
    data = []
    for fold in range(5):
        inputs, roles = load_saved_fold(args.study, fold, membership)
        gate, gate_provenance = load_gate_training(args.learned, fold, roles, membership)
        data.append((inputs, gate, roles, gate_provenance))
    args.out.mkdir(parents=True)
    code_dir = args.out / "code"
    code_dir.mkdir()
    code_files = [Path(__file__)] + [Path(__file__).with_name(name) for name in (
        "learned_budget.py", "budget.py", "run_rare_class.py", "rare_class.py", "learned_policy.py", "policy.py")]
    for path in code_files:
        (code_dir / path.name).write_bytes(path.read_bytes())
    protocol = REPO / "docs/uq_research/aris/refine-logs/EXPERIMENT_PLAN.md"
    if protocol.is_file():
        (code_dir / "EXPERIMENT_PLAN.md").write_bytes(protocol.read_bytes())
    provenance = dict(
        interpretation="Posthoc exploratory checkpoint playback on reused development objects; no confirmatory claim",
        protocol_sha256=sha(protocol) if protocol.is_file() else None,
        source_study=str(args.study.resolve()), source_learned=str(args.learned.resolve()),
        source_study_results_sha256=source_hash, source_learned_results_sha256=learned_hash,
        source_membership_sha256=sha(args.study / "membership.csv"),
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        code_sha256={p.name: sha(p) for p in code_files}, classes=CLASSES, horizons=HORIZONS,
        package_versions={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")},
        alphas=ALPHAS, regimes=["pooled", "mondrian"], budget_multipliers=BUDGET_MULTIPLIERS,
        heuristic_policies=HEURISTICS, random_seeds=RANDOM_SEEDS, learned_variants=VARIANTS,
        continuation_training_role="proper-training early-stop objects only; each eligible object total fitting weight one",
        classifier_score_dependency="gate predictions were used for classifier early stopping; not untouched out-of-fold scores",
        target_threshold_dependency="calibration-label-derived thresholds determine training set closures; features/target truth use gate objects only",
        feature_label_source="probability_argmax for all continuation models; original hierarchy labels unused",
        score="max(predicted reward,0)/clip(predicted query cost,1,3)", ridge_alpha=1,
        initial_query_cost="one day5 checkpoint per object, reported separately",
        extra_query_cost="one token reveals ONE next scheduled checkpoint; no future nobs for allocation",
        closure="first singleton automatic; empty referral; terminal multiple exhausted; otherwise budget unresolved",
        winner_selection=False, classifier_retrained=False, test_outcomes_read=False, anomaly_rows_used=False)
    write_json(args.out / "provenance.json", provenance)
    folds, tick = [], time.monotonic()
    for fold, (inputs, gate, roles, gate_provenance) in enumerate(data):
        folder = args.out / f"fold{fold}"
        result = dict(fold=fold, **run_budget_fold(inputs, gate, folder))
        write_json(folder / "provenance.json", dict(**roles, **gate_provenance))
        write_json(folder / "result.json", result)
        folds.append(result)
        print(f"fold={fold} saved {len(result['records'])} allocation cells; elapsed={time.monotonic()-tick:.3f}s", flush=True)
    pooled = aggregate_budget_folds(folds)
    write_compact(args.out, pooled)
    artifacts = {str(p.relative_to(args.out)): sha(p) for p in sorted(args.out.rglob("*")) if p.is_file()}
    write_json(args.out / "results.json", dict(provenance=provenance, folds=folds, pooled=pooled,
                                               elapsed_seconds=time.monotonic()-tick, artifacts_sha256=artifacts))
    (args.out / "COMPLETE").write_text(sha(args.out / "results.json") + "\n")


if __name__ == "__main__":
    main()
