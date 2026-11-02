"""Synthetic-only checks for posthoc reliability rankers and source-role guards."""
import hashlib
import json

import numpy as np
import pytest

from scripts.uq_anytime.learned_policy import (
    HORIZONS, certify_family, first_stop, fit_rankers, full_policy_grid,
    reliability_features, score_rankers, validate_metadata,
    validate_study_complete,
)


def metadata():
    return {h: dict(training_ids=["t0", "t1"], early_stop_ids=["g0", "g1"], features=["x", "n_obs_total"])
            for h in HORIZONS}


def synthetic(n=20):
    rng = np.random.default_rng(20261102)
    probs = rng.dirichlet(np.ones(6), size=(n, 4))
    labels = probs.argmax(axis=2)
    nobs = np.broadcast_to([5, 10, 20, 40], (n, 4)).copy()
    y = np.arange(n) % 6
    return probs, nobs, labels, y


def test_role_validation_accepts_and_preserves_gate_order():
    meta = metadata()
    meta[10]["early_stop_ids"] = ["g1", "g0"]
    roles = validate_metadata(meta, ["c0"], ["e0"])
    assert roles["gate_training_ids"] == ["g0", "g1"]
    assert not set(roles["gate_training_ids"]) & (set(roles["calibration_ids"]) | set(roles["evaluation_ids"]))


def test_source_complete_marker_required_and_hash_verified(tmp_path):
    with pytest.raises(ValueError, match="incomplete"):
        validate_study_complete(tmp_path)
    result = tmp_path / "results.json"
    result.write_text('{"synthetic": true}\n')
    with pytest.raises(ValueError, match="incomplete"):
        validate_study_complete(tmp_path)
    marker = tmp_path / "COMPLETE"
    marker.write_text("done\n")
    with pytest.raises(ValueError, match="SHA256"):
        validate_study_complete(tmp_path)
    digest = hashlib.sha256(result.read_bytes()).hexdigest()
    marker.write_text(digest + "\n")
    assert validate_study_complete(tmp_path) == digest
    result.write_text('{"synthetic": false}\n')
    with pytest.raises(ValueError, match="SHA256"):
        validate_study_complete(tmp_path)


@pytest.mark.parametrize("role", ["calibration", "evaluation", "training", "cross_horizon", "duplicate", "features", "missing_horizon"])
def test_role_validation_rejects_contamination(role):
    meta = metadata()
    cal, ev = ["c0"], ["e0"]
    if role == "calibration":
        cal = ["g0"]
    elif role == "evaluation":
        ev = ["g1"]
    elif role == "training":
        meta[5]["early_stop_ids"] = ["t0", "g1"]
    elif role == "cross_horizon":
        meta[20]["early_stop_ids"] = ["c0", "g1"]
    elif role == "duplicate":
        meta[5]["early_stop_ids"] = ["g0", "g0"]
    elif role == "features":
        meta[100]["features"] = ["n_obs_total", "x"]
    else:
        del meta[100]
    with pytest.raises(ValueError):
        validate_metadata(meta, cal, ev)


def test_calibration_evaluation_overlap_is_rejected():
    with pytest.raises(ValueError, match="calibration/evaluation"):
        validate_metadata(metadata(), ["c0"], ["c0"])


def test_features_hierarchy_probability_and_causal_temporal_changes():
    probs, nobs, labels, _ = synthetic()
    labels[:] = 5  # must use actual hierarchy label, not argmax confidence
    base, names = reliability_features(probs, nobs, labels)
    assert base.shape == (20, 4, 15)
    np.testing.assert_array_equal(base[..., names.index("predicted_label_probability")], probs[..., 5])
    extended, extra_names = reliability_features(probs, nobs, labels, quality_temporal=True)
    assert extended.shape[-1] == 19
    assert np.all(extended[:, 0, extra_names.index("previous_look_probability_l1")] == 0)
    assert np.all(extended[:, 0, extra_names.index("previous_look_label_switch")] == 0)
    changed_probs, changed_labels, changed_nobs = probs.copy(), labels.copy(), nobs.copy()
    changed_probs[:, 2:] = 1 / 6
    changed_labels[:, 2:] = 1
    changed_nobs[:, 2:] = 1000
    changed, _ = reliability_features(changed_probs, changed_nobs, changed_labels, quality_temporal=True)
    np.testing.assert_array_equal(changed[:, :2], extended[:, :2])
    assert np.all(changed[:, 2, extra_names.index("previous_look_label_switch")] == 1)


def test_first_stop_ignores_future_and_honors_earliest():
    scores = np.array([[0.2, 0.8, 0.9, 0.95], [0.1, 0.2, 0.3, 0.4]])
    labels = np.array([[0, 1, 2, 3], [4, 4, 4, 4]])
    out = first_stop(scores, labels, 0.8, 5)
    assert out["stop_index"].tolist() == [1, -1]
    assert out["prediction"].tolist() == [1, -1]
    scores[0, 2:], labels[0, 2:] = 0, 5
    changed = first_stop(scores, labels, 0.8, 5)
    for key in out:
        np.testing.assert_equal(changed[key], out[key])
    assert not first_stop(scores, labels, 0.8, 20)["selected"].any()


def test_fixed_rankers_train_with_one_total_weight_per_object_and_causal_scoring():
    probs, nobs, labels, y = synthetic(30)
    fitted = fit_rankers(probs, nobs, labels, y)
    for name, (pipeline, feature_names) in fitted.items():
        assert pipeline.named_steps["ranker"].C == 1
        assert pipeline.named_steps["ranker"].max_iter == 2000
        assert pipeline.named_steps["scaler"].n_samples_seen_ == pytest.approx(30)
        assert len(feature_names) == (15 if name == "output_only" else 19)
    scores = score_rankers(fitted, probs, nobs, labels)
    changed_probs, changed_labels = probs.copy(), labels.copy()
    changed_probs[:, 3] = 1 / 6
    changed_labels[:, 3] = 0
    changed = score_rankers(fitted, changed_probs, nobs, changed_labels)
    for name in scores:
        assert scores[name].shape == (30, 4)
        np.testing.assert_array_equal(changed[name][:, :3], scores[name][:, :3])


def test_certification_full_family_shared_correction_and_separate_winners():
    n = 250
    probs = np.zeros((n, 4, 6)); probs[..., 0] = 0.99; probs[..., 1] = 0.01
    labels, nobs, y = np.zeros((n, 4), int), np.full((n, 4), 100), np.zeros(n, int)
    scores = dict(output_only=np.full((n, 4), 0.99), quality_temporal=np.full((n, 4), 0.99))
    result = certify_family(probs, nobs, labels, y, scores)
    assert len(full_policy_grid()) == result["multiplicity"] == 75
    assert result["per_policy_delta"] == pytest.approx(0.05 / 5 / 75)
    assert {r["per_policy_delta"] for r in result["records"]} == {result["per_policy_delta"]}
    assert result["winner_indices"] == dict(confidence_only=0, support_family=0, output_only=45, quality_temporal=60)
    assert "winner_index" not in result  # no mixed-family cherry-picked winner
    json.dumps(result, allow_nan=False)


def test_no_selection_returns_four_abstain_policies():
    probs = np.full((10, 4, 6), 1 / 6)
    labels, nobs, y = np.zeros((10, 4), int), np.zeros((10, 4)), np.zeros(10, int)
    result = certify_family(probs, nobs, labels, y,
                            dict(output_only=np.zeros((10, 4)), quality_temporal=np.zeros((10, 4))))
    assert all(index is None for index in result["winner_indices"].values())
    assert all(r["risk_upper_bound"] is None for r in result["records"])


def test_invalid_labels_scores_and_training_population():
    probs, nobs, labels, y = synthetic()
    with pytest.raises(ValueError):
        first_stop(np.full((2, 4), np.nan), np.zeros((2, 4), int), 0.5, 5)
    with pytest.raises(ValueError):
        reliability_features(probs, nobs, labels + 6)
    with pytest.raises(ValueError):
        fit_rankers(probs, nobs, np.zeros_like(labels), np.zeros(len(y), int))
    with pytest.raises(ValueError):
        certify_family(probs, nobs, labels, y, dict(output_only=np.zeros((20, 4))))
