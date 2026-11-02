"""Training-role, target, causal-feature and frozen-runner checks using toys only."""
import json

import numpy as np
import pytest

from scripts.uq_anytime.budget import PrefixObservation, allocate_query_budget
from scripts.uq_anytime.learned_budget import (
    VARIANTS, continuation_features, continuation_scorer, fit_continuation_models, model_record, training_targets,
)
from scripts.uq_anytime.learned_policy import sha
from scripts.uq_anytime.rare_class import CLASSES, HORIZONS
from scripts.uq_anytime.run_budget import (
    ALPHAS, BUDGET_MULTIPLIERS, HEURISTICS, RANDOM_SEEDS, aggregate_budget_folds,
    load_gate_training, prediction_diagnostics, run_budget_fold, summarize_allocation,
)


def paths():
    sets = np.ones((4, 4, 6), dtype=bool)
    sets[0, 2:] = False
    sets[0, 2:, 0] = True
    sets[0, 3] = False  # future empty cannot revoke first correct singleton
    sets[1, 1:] = False
    sets[1, 1:, 5] = True  # wrong singleton closes before further opportunities
    sets[2] = False  # immediate referral creates no examples
    y = np.array([0, 1, 2, 0])
    return sets, y


def training_arrays():
    rng = np.random.default_rng(42)
    probs = rng.dirichlet(np.ones(6), size=(4, 4))
    nobs = np.broadcast_to([1, 3, 7, 11], (4, 4)).copy()
    labels = np.zeros((4, 4), dtype=int)
    return probs, nobs, labels


def test_targets_use_first_future_closure_and_equal_object_weight():
    sets, y = paths()
    t = training_targets(sets, y)
    assert np.array_equal(t["object_index"], [0, 0, 1, 3, 3, 3])
    assert np.array_equal(t["look_index"], [0, 1, 0, 0, 1, 2])
    assert np.array_equal(t["reward_aggregate"], [1, 1, 0, 0, 0, 0])
    assert np.allclose(t["reward_macro"], [1/3, 1/3, 0, 0, 0, 0])
    assert np.array_equal(t["cost"], [2, 1, 1, 3, 2, 1])
    for obj in np.unique(t["object_index"]):
        assert t["sample_weight"][t["object_index"] == obj].sum() == pytest.approx(1)
    assert t["sample_weight"].sum() == pytest.approx(3)


def test_future_empty_has_no_reward_and_terminal_multiple_costs_all_remaining_queries():
    sets = np.ones((2, 4, 6), bool)
    sets[0, 2:] = False
    t = training_targets(sets, np.array([0, 1]))
    assert not t["reward_aggregate"].any()
    assert np.array_equal(t["cost"], [2, 1, 3, 2, 1])


def test_training_features_use_argmax_and_append_known_look_index_causally():
    probs, nobs, _ = training_arrays()
    for quality in (False, True):
        full, names = continuation_features(probs, nobs, quality_temporal=quality)
        assert names[-1] == "current_look_index"
        assert np.array_equal(full[..., -1], np.broadcast_to(np.arange(4), (4, 4)))
        for length in (1, 2, 3):
            prefix, prefix_names = continuation_features(probs[:, :length], nobs[:, :length], quality_temporal=quality)
            assert prefix_names == names
            assert np.array_equal(prefix, full[:, :length])


def test_fits_ignore_original_hierarchy_labels_and_fit_with_equal_object_weight():
    sets, y = paths()
    probs, nobs, labels = training_arrays()
    a = fit_continuation_models(probs, nobs, labels, y, sets)
    b = fit_continuation_models(probs, nobs, np.full_like(labels, 5), y, sets)
    t = training_targets(sets, y)
    for name in VARIANTS:
        assert model_record(a[name]) == model_record(b[name])
        record = a[name]
        assert record["ridge_alpha"] == 1
        assert record["n_training_rows"] == 6
        assert record["n_eligible_objects"] == 3
        assert record["sum_sample_weights"] == pytest.approx(3)
        features, _ = continuation_features(probs, nobs, quality_temporal=name == "macro_quality")
        X = features[t["object_index"], t["look_index"]]
        expected_mean = np.average(X, axis=0, weights=t["sample_weight"])
        assert np.allclose(record["reward_model"].named_steps["scaler"].mean_, expected_mean)
        assert np.allclose(record["cost_model"].named_steps["scaler"].mean_, expected_mean)
        json.dumps(model_record(record), allow_nan=False)


@pytest.mark.parametrize("no_eligible", [False, True])
def test_no_positive_reward_is_explicit_zero_fallback(no_eligible):
    probs, nobs, labels = training_arrays()
    sets = np.ones_like(probs, dtype=bool)
    if no_eligible:
        sets[:] = False
    fitted = fit_continuation_models(probs, nobs, labels, np.array([0, 1, 2, 3]), sets)
    for record in fitted.values():
        assert record["reward_model"] is None
        assert record["constant_reward"] == 0
        assert record["no_positive_reward"]
        assert record["no_eligible_prefixes"] == no_eligible
        scorer = continuation_scorer(record)
        observation = PrefixObservation(0, "toy", 0, sets[0, :1], probs[0, :1], nobs[0, :1])
        assert scorer(observation) == 0
        if no_eligible:
            assert record["cost_model"] is None and record["constant_cost"] == 1


class ConstantPrediction:
    def __init__(self, value):
        self.value = value

    def predict(self, values):
        return np.full(len(values), self.value)


@pytest.mark.parametrize("reward,cost,expected", [(-2, 2, 0), (2, .2, 2), (2, 10, 2/3)])
def test_score_clips_reward_and_expected_query_cost_without_future_inputs(reward, cost, expected):
    probs, nobs, _ = training_arrays()
    sets, _ = paths()
    _, names = continuation_features(probs, nobs)
    record = dict(constant_reward=None, constant_cost=None, reward_model=ConstantPrediction(reward),
                  cost_model=ConstantPrediction(cost), quality_temporal=False, feature_names=names)
    scorer = continuation_scorer(record)
    observation = PrefixObservation(0, "toy", 1, sets[0, :2], probs[0, :2], nobs[0, :2])
    assert scorer(observation) == pytest.approx(expected)
    # The memoized result is for this already-exposed prefix, never future data.
    assert scorer(observation) == pytest.approx(expected)


def test_true_class_error_and_emitted_label_precision_have_distinct_denominators():
    predictions = np.array([4, 4, 2, -1])
    y = np.array([4, 0, 0, 2])
    diagnostic = prediction_diagnostics(predictions, y, predictions >= 0)
    assert diagnostic["by_predicted_class"]["TDE"] == dict(n_emitted=2, n_correct=1, n_wrong=1, precision=.5, error_given_emitted=.5)
    assert diagnostic["confusion_true_rows_predicted_columns"][0][4] == 1
    assert diagnostic["by_predicted_class"]["CV"]["precision"] is None


def toy_inputs():
    rng = np.random.default_rng(91)
    nc, ne, ng = 120, 12, 24
    ids = np.array([f"cal{i}" for i in range(nc)] + [f"eval{i}" for i in range(ne)])
    y = np.arange(nc + ne) % 6
    probs = rng.dirichlet(np.ones(6), size=(nc + ne, 4))
    inputs = dict(obj_id=ids, calibration_ids=ids[:nc], evaluation_ids=ids[nc:], probabilities=probs,
                  nobs=np.broadcast_to([1, 2, 4, 8], (nc + ne, 4)), hierarchy_labels=np.zeros((nc + ne, 4), int),
                  calibration_y=y[:nc], evaluation_y=y[nc:])
    gate = dict(obj_id=np.array([f"train{i}" for i in range(ng)]),
                probabilities=rng.dirichlet(np.ones(6), size=(ng, 4)),
                nobs=np.broadcast_to([1, 2, 4, 8], (ng, 4)), labels=np.zeros((ng, 4), int), y=np.arange(ng) % 6)
    return inputs, gate


def test_allocation_summary_accounts_initial_extra_unused_tokens_and_early_referrals():
    sets, y = paths()
    probs, nobs, _ = training_arrays()
    allocation = allocate_query_budget(sets, probs, nobs, ["a", "b", "c", "d"], 12)
    metrics = summarize_allocation(sets, nobs, allocation, y, 12)
    assert metrics["resources"]["initial_checkpoint_queries"] == 4
    assert metrics["resources"]["consumed_extra_tokens"] == int(allocation.final_look.sum())
    assert metrics["resources"]["unused_extra_tokens"] == 12-allocation.consumed_tokens
    assert metrics["status_counts"] == dict(automatic=2, empty=1, exhausted=1, budget=0)
    # Empty initial object consumes only the initial count, not terminal counts.
    expected_nobs = nobs[np.arange(4), allocation.final_look].sum()
    assert metrics["overall"]["total_nobs_consumed"] == expected_nobs


def test_toy_budget_runner_saves_every_frozen_cell_and_reproducible_counts(tmp_path):
    inputs, gate = toy_inputs()
    result = run_budget_fold(inputs, gate, tmp_path / "pilot")
    expected = 2 * len(ALPHAS) * len(BUDGET_MULTIPLIERS) * (len(HEURISTICS)+len(RANDOM_SEEDS)+len(VARIANTS))
    assert len(result["records"]) == expected
    for record in result["records"]:
        path = tmp_path / "pilot" / record["decision_artifact"]
        assert sha(path) == record["decision_sha256"]
        with np.load(path, allow_pickle=False) as saved:
            selected = saved["status"] == "automatic"
            wrong = selected & (saved["prediction"] != inputs["evaluation_y"])
            assert int(selected.sum()) == record["metrics"]["overall"]["n_selected"]
            assert int(wrong.sum()) == record["metrics"]["overall"]["n_wrong_automatic"]
            assert int(saved["final_look"].sum()) == len(saved["action_token"])
            assert record["metrics"]["resources"]["consumed_extra_tokens"] == len(saved["action_token"])
            assert np.array_equal(saved["evaluation_ids"], inputs["evaluation_ids"])
            if record["budget_multiplier"] == 0:
                assert not saved["final_look"].any()
    pooled = aggregate_budget_folds([result, result])
    assert len(pooled) == expected
    assert pooled[0]["metrics"]["overall"]["n_objects"] == 24
    assert pooled[0]["metrics"]["resources"]["initial_checkpoint_queries"] == 24
    for regime in ("pooled", "mondrian"):
        for alpha in ALPHAS:
            for budget in (0, 3):
                records = [r for r in result["records"] if r["regime"] == regime and r["alpha"] == alpha
                           and r["budget_multiplier"] == budget]
                # With no extra tokens or enough to reveal every unclosed path,
                # scheduler choice cannot change any all-object final outcome.
                with np.load(tmp_path / "pilot" / records[0]["decision_artifact"]) as saved:
                    reference = {key: saved[key].copy() for key in ("prediction", "status", "final_look")}
                for record in records[1:]:
                    with np.load(tmp_path / "pilot" / record["decision_artifact"]) as saved:
                        for key, values in reference.items():
                            assert np.array_equal(saved[key], values)
    metadata = json.loads((tmp_path / "pilot" / "model_metadata.json").read_text())
    assert len(metadata) == 4
    assert set(metadata["envelope_mondrian_a0"]["models"]) == set(VARIANTS)


def test_gate_loader_rejects_checksum_role_and_dataset_truth_contamination(tmp_path):
    _, gate = toy_inputs()
    folder = tmp_path / "fold0"
    folder.mkdir()
    source = folder / "gate_training.npz"
    np.savez(source, **gate)
    (folder / "provenance.json").write_text(json.dumps(dict(gate_training_sha256=sha(source))))
    roles = dict(gate_training_ids=gate["obj_id"].tolist(), calibration_ids=["cal"], evaluation_ids=["eval"])
    membership = {obj: (2, int(y)) for obj, y in zip(gate["obj_id"], gate["y"])}
    loaded, provenance = load_gate_training(tmp_path, 0, roles, membership)
    assert np.array_equal(loaded["obj_id"], gate["obj_id"])
    assert provenance["continuation_training_ids"] == gate["obj_id"].tolist()
    with pytest.raises(ValueError, match="IDs/order"):
        load_gate_training(tmp_path, 0, dict(roles, gate_training_ids=roles["gate_training_ids"][::-1]), membership)
    with pytest.raises(ValueError, match="roles contaminated"):
        load_gate_training(tmp_path, 0, dict(roles, evaluation_ids=[gate["obj_id"][0]]), membership)
    with pytest.raises(ValueError, match="truth differs"):
        load_gate_training(tmp_path, 0, roles, dict(membership, train0=(2, 5)))
    source.write_bytes(source.read_bytes()+b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_gate_training(tmp_path, 0, roles, membership)


def test_evaluation_truth_changes_metrics_only_never_models_or_query_actions(tmp_path):
    inputs, gate = toy_inputs()
    first = run_budget_fold(inputs, gate, tmp_path / "first")
    changed = dict(inputs, evaluation_y=(inputs["evaluation_y"]+1) % 6,
                   hierarchy_labels=np.full_like(inputs["hierarchy_labels"], 5))
    second = run_budget_fold(changed, gate, tmp_path / "second")
    assert (tmp_path / "first" / "model_metadata.json").read_bytes() == (tmp_path / "second" / "model_metadata.json").read_bytes()
    for a, b in zip(first["records"], second["records"]):
        assert a["decision_artifact"] == b["decision_artifact"]
        with np.load(tmp_path / "first" / a["decision_artifact"]) as av, np.load(tmp_path / "second" / b["decision_artifact"]) as bv:
            for key in av.files:
                if av[key].dtype.kind in "fc":
                    assert np.array_equal(av[key], bv[key], equal_nan=True)
                else:
                    assert np.array_equal(av[key], bv[key])


def test_all_variants_share_identical_known_schedule_feature_basis():
    probs, nobs, _ = training_arrays()
    output, output_names = continuation_features(probs, nobs)
    quality, quality_names = continuation_features(probs, nobs, quality_temporal=True)
    assert output_names.count("log1p_horizon") == 1
    assert quality_names.count("log1p_horizon") == 1
    assert set(quality_names) - set(output_names) == {
        "log1p_nobs", "previous_look_probability_l1", "previous_look_label_switch"}
    for name in output_names:
        assert np.array_equal(output[..., output_names.index(name)], quality[..., quality_names.index(name)])
