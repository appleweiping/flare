"""Synthetic/exhaustive checks for the exploratory class-set/budget pilot."""
import json

import numpy as np
import pytest

from scripts.uq_anytime.rare_class import (
    ALPHAS, ARMS, CLASSES, HORIZONS, calibrate_sets, finite_sample_quantile,
    first_singleton, nested_candidate_sets, summarize_sets,
)
from scripts.uq_anytime.run_rare_class import (
    _aggregate_groups, aggregate_folds, calibration_record, load_saved_fold, run_fold,
)
from scripts.uq_anytime.learned_policy import sha


def test_finite_rank_and_ties_never_clip_to_observed_max():
    assert finite_sample_quantile([], .2) == np.inf
    assert finite_sample_quantile([.1, .3, .3, .9], .2) == .9
    assert finite_sample_quantile([.1, .3, .3, .9], .4) == .3
    assert finite_sample_quantile([.1, .3, .3, .9], .19) == np.inf
    # Exact rank boundary, not a floating point interpolating quantile.
    assert finite_sample_quantile(np.arange(19), .05) == 18
    for bad in (0, 1, np.nan, True, [.1]):
        with pytest.raises(ValueError):
            finite_sample_quantile([1, 2], bad)


def test_empty_mondrian_class_keeps_candidate_and_reports_resolution():
    probs = np.array([[[.9, .1], [.8, .2]], [[.8, .2], [.9, .1]]])
    cal = calibrate_sets(probs, [0, 0], .4)
    assert np.array_equal(cal["n_calibration_by_true_class"], [2, 0])
    assert np.array_equal(cal["n_calibration_for_threshold"], [2, 0])
    assert np.allclose(cal["pvalue_resolution"], [1/3, 1])
    assert np.isinf(cal["thresholds"][:, 1]).all()
    assert nested_candidate_sets(probs, cal)[:, :, 1].all()
    pooled = calibrate_sets(probs, [0, 0], .4, mondrian=False)
    assert np.array_equal(pooled["n_calibration_for_threshold"], [2, 2])
    # A strict JSON record preserves the distinction between +inf and missing.
    record = calibration_record(cal)
    json.dumps(record, allow_nan=False)
    assert record["thresholds"][0][1] is None
    assert record["threshold_positive_infinity"][0][1]


@pytest.mark.parametrize("mondrian", [False, True])
@pytest.mark.parametrize("method", ["envelope", "bonferroni", "terminal", "fixedlook"])
def test_exhaustive_exchangeable_query_assignments_within_each_class(method, mondrian):
    rng = np.random.default_rng(71)
    # A finite population of 16 objects per class, four dependent looks each.
    # Exchangeability is enumerated by every possible held-out query assignment;
    # calibration order does not affect these symmetric order statistics.
    probs = rng.dirichlet([.5, .5], size=(32, 4))
    y = np.repeat([0, 1], 16)
    for alpha in (.25, .5):
        excluded_by_class = [[], []]
        for query in range(len(probs)):
            mask = np.arange(len(probs)) != query
            cal = calibrate_sets(probs[mask], y[mask], alpha, method=method, mondrian=mondrian,
                                 **({"look_index": 1} if method == "fixedlook" else {}))
            path = nested_candidate_sets(probs[[query]], cal)
            excluded_by_class[y[query]].append(not path[0, :, y[query]].all())
        if mondrian:
            assert all(np.mean(v) <= alpha for v in excluded_by_class)
        else:
            assert np.mean(excluded_by_class) <= alpha


def test_envelope_max_score_threshold_and_bonferroni_fixed_four_looks():
    rng = np.random.default_rng(13)
    probs = rng.dirichlet(np.ones(3), size=(80, 4))
    y = np.arange(80) % 3
    for method in ("envelope", "bonferroni"):
        cal = calibrate_sets(probs, y, .2, method=method)
        q = cal["thresholds"]
        for c in range(3):
            truth_scores = 1 - probs[y == c, :, c]
            for t in range(4):
                scores = truth_scores.max(axis=1) if method == "envelope" else truth_scores[:, t]
                alpha = .2 if method == "envelope" else .2 / 4
                assert q[t, c] == finite_sample_quantile(scores, alpha)
        sets = nested_candidate_sets(probs, cal)
        assert not np.any(sets[:, 1:] & ~sets[:, :-1])
        if method == "envelope":
            assert np.array_equal(sets, np.maximum.accumulate(1 - probs, axis=1) <= q[None])
        # Tail probabilities cannot alter prefixes or thresholds.
        changed = probs.copy()
        changed[:, 2:] = [1, 0, 0]
        assert np.array_equal(nested_candidate_sets(changed, cal)[:, :2], sets[:, :2])


def manual_paths():
    # First object becomes a singleton then empty. Third becomes singleton late.
    sets = np.array([[[1, 0], [0, 0], [0, 0], [0, 0]],
                     [[0, 0], [0, 0], [0, 0], [0, 0]],
                     [[1, 1], [1, 1], [0, 1], [0, 1]]], bool)
    nobs = np.broadcast_to([2, 5, 9, 25], (3, 4)).copy()
    return sets, nobs


def test_first_singleton_retained_empty_abstains_and_all_objects_pay_resources():
    sets, nobs = manual_paths()
    decision = first_singleton(sets, nobs)
    assert np.array_equal(decision["selected"], [True, False, True])
    assert np.array_equal(decision["prediction"], [0, -1, 1])
    assert np.array_equal(decision["stop_index"], [0, 3, 2])
    assert np.array_equal(decision["decision_index"], [0, -1, 2])
    assert np.array_equal(decision["consumed_day"], [5, 100, 20])
    assert np.array_equal(decision["nobs_consumed"], [2, 25, 9])
    early = first_singleton(sets, nobs, budget_day=10)
    assert np.array_equal(early["prediction"], [0, -1, -1])
    assert np.array_equal(early["stop_index"], [0, 1, 1])
    assert np.array_equal(early["nobs_consumed"], [2, 5, 5])
    # A changed later nested path cannot revoke or relabel the first decision.
    changed = sets.copy()
    changed[0, 1:, 0] = True
    other = first_singleton(changed, nobs)
    assert decision["prediction"][0] == other["prediction"][0]
    assert decision["stop_index"][0] == other["stop_index"][0]


def test_metrics_use_dataset_truth_and_both_error_denominators():
    sets, nobs = manual_paths()
    decision = first_singleton(sets, nobs)
    y = np.array([1, 0, 1])
    metrics = summarize_sets(sets, decision, y, class_names=("a", "b"))
    overall = metrics["overall"]
    assert overall["n_wrong_automatic"] == 1
    assert overall["n_correct_automatic"] == 1
    assert overall["n_unresolved"] == 1
    assert overall["mean_set_size_at_consumed"] == pytest.approx(2/3)
    assert overall["mean_set_size_unresolved"] == 0
    assert overall["n_true_class_retained_unresolved"] == 0
    assert overall["wrong_automatic_per_all"] == pytest.approx(1/3)
    assert overall["selected_error"] == .5
    assert overall["selection_coverage"] == pytest.approx(2/3)
    assert overall["true_class_lifetime_exclusion"] == pytest.approx(2/3)
    assert overall["mean_day_selected"] == 12.5
    assert overall["mean_day_consumed_all"] == pytest.approx(125/3)
    assert overall["total_nobs_consumed"] == 36  # NOT sum of cumulative nobs at all looks
    assert metrics["by_true_class"]["a"]["selected_error"] is None
    assert metrics["by_true_class"]["b"]["n_wrong_automatic"] == 1
    # With truth0, first decision is correct but a future set loses truth.
    diagnostic = summarize_sets(sets, decision, np.array([0, 0, 1]), class_names=("a", "b"))["overall"]
    assert diagnostic["n_true_class_lifetime_excluded"] == 2
    assert diagnostic["n_true_class_consumed_prefix_excluded"] == 1
    assert diagnostic["n_wrong_automatic"] == 0


def test_terminal_is_unrestricted_before_terminal_and_early_budgets_abstain():
    probs = np.tile([[[.9, .1]]], (20, 4, 1))
    cal = calibrate_sets(probs, np.zeros(20, dtype=int), .2, method="terminal", mondrian=False)
    sets = nested_candidate_sets(probs, cal)
    assert sets[:, :3].all()
    nobs = np.broadcast_to([1, 2, 4, 8], (20, 4))
    for budget in HORIZONS[:-1]:
        decision = first_singleton(sets, nobs, budget_day=budget)
        assert not decision["selected"].any()
        assert (decision["consumed_day"] == budget).all()
    decision = first_singleton(sets, nobs)
    assert decision["selected"].all()
    assert (decision["consumed_day"] == 100).all()


def test_reject_non_nested_paths_and_non_cumulative_costs():
    sets, nobs = manual_paths()
    sets[0, -1, 1] = True
    with pytest.raises(ValueError, match="nested"):
        first_singleton(sets, nobs)
    sets, nobs = manual_paths()
    nobs[:, -1] = 0
    with pytest.raises(ValueError, match="cumulative"):
        first_singleton(sets, nobs)
    with pytest.raises(ValueError, match="horizon"):
        first_singleton(sets, np.ones((3, 4)), budget_day=15)
    for y in ([0., 1., 0.], [-1, 0, 1]):
        with pytest.raises(ValueError, match="integer true class"):
            calibrate_sets(np.full((3, 4, 2), .5), y, .1)


def test_empty_calibration_and_empty_query_are_explicit():
    cal = calibrate_sets(np.empty((0, 4, 6)), np.array([], dtype=int), .1)
    assert np.isposinf(cal["thresholds"]).all()
    sets = nested_candidate_sets(np.empty((0, 4, 6)), cal)
    decision = first_singleton(sets, np.empty((0, 4)))
    metrics = summarize_sets(sets, decision, np.array([], dtype=int))
    assert metrics["overall"]["n_objects"] == 0
    assert metrics["overall"]["selected_error"] is None


def synthetic_inputs():
    rng = np.random.default_rng(31)
    nc, ne = 120, 6
    ids = np.array([f"obj{i}" for i in range(nc + ne)])
    y = np.arange(nc + ne) % 6
    probs = rng.dirichlet(np.ones(6), size=(nc + ne, 4))
    return dict(obj_id=ids, calibration_ids=ids[:nc], evaluation_ids=ids[nc:],
                probabilities=probs, nobs=np.broadcast_to([1, 3, 6, 18], (nc + ne, 4)),
                hierarchy_labels=np.zeros((nc + ne, 4), dtype=int), calibration_y=y[:nc], evaluation_y=y[nc:])


def test_toy_fold_roundtrip_all_arms_budgets_and_no_truth_in_decisions(tmp_path):
    inputs = synthetic_inputs()
    result = run_fold(inputs, tmp_path / "first")
    assert len(result["records"]) == len(ARMS) * len(ALPHAS) * len(HORIZONS)
    # Corrupt BOTH evaluation truth and hierarchy labels: set/decision outputs
    # cannot change; only metrics against dataset truth can change.
    changed = dict(inputs, evaluation_y=(inputs["evaluation_y"] + 1) % 6,
                   hierarchy_labels=np.full_like(inputs["hierarchy_labels"], 5))
    run_fold(changed, tmp_path / "second")
    with np.load(tmp_path / "first" / "sets_decisions.npz") as a, np.load(tmp_path / "second" / "sets_decisions.npz") as b:
        assert set(a.files) == set(b.files)
        for key in a.files:
            assert np.array_equal(a[key], b[key], equal_nan=True) if a[key].dtype.kind != "U" else np.array_equal(a[key], b[key])
        for record in result["records"]:
            key = record["calibration_key"]
            decision = {field: a[f"{key}_b{record['budget_day']}_{field}"] for field in (
                "selected", "prediction", "stop_index", "consumed_day", "nobs_consumed")}
            actual = summarize_sets(a[f"{key}_candidate_sets"], decision, inputs["evaluation_y"])
            assert actual == record["metrics"]
    pooled = aggregate_folds([result, result])
    assert pooled[0]["metrics"]["overall"]["n_objects"] == 12
    assert pooled[0]["metrics"]["overall"]["mean_day_consumed_all"] == result["records"][0]["metrics"]["overall"]["mean_day_consumed_all"]
    with pytest.raises(FileExistsError):
        run_fold(inputs, tmp_path / "first")


def test_aggregation_weights_counts_instead_of_fold_rates():
    sets, nobs = manual_paths()
    decision = first_singleton(sets, nobs)
    first = summarize_sets(sets, decision, np.array([1, 0, 1]), class_names=("a", "b"))["overall"]
    second = summarize_sets(sets[:1], {k: v[:1] for k, v in decision.items()}, np.array([0]), class_names=("a", "b"))["overall"]
    pooled = _aggregate_groups([first, second])
    assert pooled["wrong_automatic_per_all"] == .25
    assert pooled["selected_error"] == pytest.approx(1/3)
    assert pooled["mean_day_selected"] == 10


def test_source_loader_checks_dataset_truth_roles_and_excludes_anomaly_rows(tmp_path):
    folder = tmp_path / "fold0"
    folder.mkdir()
    cal_ids, ev_ids = np.array(["c0", "c1"]), np.array(["e0", "e1"])
    membership = {"c0": (1, 0), "c1": (1, 1), "e0": (0, 2), "e1": (0, 3)}
    probs = np.full((5, 4, 6), 1/6)
    probs[-1] = np.nan  # excluded anomaly must not enter construction/evaluation
    arrays = dict(obj_id=np.concatenate([cal_ids, ev_ids, ["anomaly"]]), calibration_ids=cal_ids, evaluation_ids=ev_ids,
                  probabilities=probs, labels=np.zeros((5, 4), dtype=int), nobs=np.ones((5, 4)),
                  calibration_y=np.array([0, 1]), evaluation_y=np.array([2, 3]))
    np.savez(folder / "predictions.npz", **arrays)
    for h in HORIZONS:
        metadata = dict(training_ids=["train"], early_stop_ids=["early"], features=["x"])
        for part in ("top", "branch"):
            path = folder / f"h{h}_{part}.txt"
            path.write_text("synthetic model only\n")
            metadata[f"{part}_sha256"] = sha(path)
        (folder / f"h{h}_metadata.json").write_text(json.dumps(metadata))
    inputs, provenance = load_saved_fold(tmp_path, 0, membership)
    assert inputs["probabilities"].shape == (4, 4, 6)
    assert np.isfinite(inputs["probabilities"]).all()
    assert provenance["evaluation_ids"] == ["e0", "e1"]
    with pytest.raises(ValueError, match="truth differs"):
        load_saved_fold(tmp_path, 0, dict(membership, e0=(0, 5)))
    arrays["obj_id"] = np.concatenate([ev_ids, cal_ids, ["anomaly"]])
    np.savez(folder / "predictions.npz", **arrays)
    with pytest.raises(ValueError, match="ID ordering"):
        load_saved_fold(tmp_path, 0, membership)


@pytest.mark.parametrize("deadline_index", range(4))
def test_fixedlook_full_alpha_calibration_is_informed_at_each_deadline(deadline_index):
    # Both classes have 20 calibration objects. Two query types: singleton or
    # unresolved set containing truth. All objects pay the same deadline cost.
    probs = np.tile([[[.6, .4]]], (40, 4, 1))
    y = np.repeat([0, 1], 20)
    probs[y == 1] = [.4, .6]
    cal = calibrate_sets(probs, y, .1, method="fixedlook", look_index=deadline_index)
    assert cal["effective_alpha"] == .1
    assert cal["calibration_look_index"] == deadline_index
    assert np.isposinf(np.delete(cal["thresholds"], deadline_index, axis=0)).all()
    query = np.array([[[.9, .1]] * 4, [[.5, .5]] * 4])
    # Use a larger threshold only for class1 to make second object unresolved.
    cal["thresholds"][deadline_index, 1] = .6
    query[1] = [.6, .4]
    sets = nested_candidate_sets(query, cal)
    assert sets[:, :deadline_index].all()
    assert np.array_equal(sets[:, deadline_index:], np.repeat(sets[:, deadline_index:deadline_index+1], 4-deadline_index, axis=1))
    nobs = np.broadcast_to([1, 2, 4, 8], (2, 4))
    decision = first_singleton(sets, nobs, budget_day=HORIZONS[deadline_index])
    assert np.array_equal(decision["selected"], [True, False])
    assert (decision["consumed_day"] == HORIZONS[deadline_index]).all()
    assert (decision["nobs_consumed"] == nobs[:, deadline_index]).all()
    # Independent of whether calibrated threshold creates a singleton, all
    # decisions at this deadline cannot happen earlier.
    assert (decision["stop_index"] == deadline_index).all()
    if deadline_index:
        assert not first_singleton(sets, nobs, budget_day=HORIZONS[deadline_index-1])["selected"].any()
    for c in range(2):
        if c == 0:
            assert cal["thresholds"][deadline_index, c] == finite_sample_quantile(1-probs[y == c, deadline_index, c], .1)


def test_unresolved_retained_truth_and_set_size_utility():
    sets = np.ones((2, 4, 2), dtype=bool)
    sets[1] = False
    nobs = np.broadcast_to([1, 2, 4, 8], (2, 4))
    decision = first_singleton(sets, nobs)
    assert np.array_equal(decision["set_size_at_stop"], [2, 0])
    metrics = summarize_sets(sets, decision, np.array([0, 1]), class_names=("a", "b"))["overall"]
    assert metrics["n_true_class_retained_unresolved"] == 1
    assert metrics["true_class_retained_given_unresolved"] == .5
    assert metrics["true_class_retained_unresolved_per_all"] == .5
    assert metrics["mean_set_size_unresolved"] == 1


def test_full_synthetic_cli_completion_manifest_and_overwrite_guard(tmp_path, monkeypatch):
    from scripts.uq_anytime.run_rare_class import main
    study = tmp_path / "study"
    study.mkdir()
    rows = [(f"obj{i}", i // 2, CLASSES[i % 6]) for i in range(10)]
    (study / "membership.csv").write_text("obj_id,fold,cls\n" + "".join(f"{obj_id},{fold},{cls}\n" for obj_id, fold, cls in rows))
    (study / "results.json").write_text('{"synthetic":true}\n')
    (study / "COMPLETE").write_text(sha(study / "results.json") + "\n")
    for fold in range(5):
        folder = study / f"fold{fold}"
        folder.mkdir()
        cal = [row for row in rows if row[1] == (fold + 1) % 5]
        ev = [row for row in rows if row[1] == fold]
        proper = [row[0] for row in rows if row[1] not in {fold, (fold + 1) % 5}]
        cal_ids, ev_ids = [r[0] for r in cal], [r[0] for r in ev]
        np.savez(folder / "predictions.npz", obj_id=np.array(cal_ids + ev_ids),
                 calibration_ids=np.array(cal_ids), evaluation_ids=np.array(ev_ids),
                 probabilities=np.full((4, 4, 6), 1/6), labels=np.zeros((4, 4), int),
                 nobs=np.broadcast_to([1, 2, 4, 8], (4, 4)),
                 calibration_y=np.array([CLASSES.index(r[2]) for r in cal]),
                 evaluation_y=np.array([CLASSES.index(r[2]) for r in ev]))
        for h in HORIZONS:
            meta = dict(training_ids=proper[:4], early_stop_ids=proper[4:], features=["x"])
            for part in ("top", "branch"):
                path = folder / f"h{h}_{part}.txt"
                path.write_text("synthetic model\n")
                meta[f"{part}_sha256"] = sha(path)
            (folder / f"h{h}_metadata.json").write_text(json.dumps(meta))
    out = tmp_path / "pilot"
    monkeypatch.setattr("sys.argv", ["run_rare_class.py", "--study", str(study), "--out", str(out)])
    main()
    assert (out / "COMPLETE").read_text().strip() == sha(out / "results.json")
    results = json.loads((out / "results.json").read_text())
    assert len(results["pooled"]) == len(ARMS) * len(ALPHAS) * len(HORIZONS)
    for relative, digest in results["artifacts_sha256"].items():
        assert sha(out / relative) == digest
    assert results["provenance"]["winner_selection"] is False
    with pytest.raises(ValueError, match="already exists"):
        main()
