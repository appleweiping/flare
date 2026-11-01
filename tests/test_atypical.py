"""Atypicality-stratified sets (flare.atypical) and the calibrated review trigger."""
import unittest
from pathlib import Path

import numpy as np

from flare import atypical as A

MODELS = Path(__file__).resolve().parents[1] / "models" / "bts6"


def toy(n, rng):
    P = rng.dirichlet(np.ones(6) * .4, n)
    y = np.array([rng.choice(6, p=p) for p in P])
    return P, y, -np.log(P.max(1)) + rng.normal(0, .1, n)


class AtypicalTests(unittest.TestCase):
    def test_unit_weight_quantile_matches_flare_conformal(self):
        from flare import conformal
        rng = np.random.default_rng(0)
        for n in (10, 11, 57, 400):
            s = rng.random(n)
            q = conformal.calibrate(np.c_[1 - s, np.zeros((n, conformal.NUM_CLASSES - 1))], np.zeros(n, int), .1)[0]
            self.assertEqual(A.weighted_quantile(s, None, .1), q)

    def test_replicated_views_count_once(self):
        rng = np.random.default_rng(1)
        for n in (5, 16, 99):
            s = rng.random(n)
            for r in (2, 4):
                obj = np.tile(np.arange(n), r)
                self.assertEqual(A.weighted_quantile(np.tile(s, r), A.pool_weights(obj)), A.weighted_quantile(s))

    def test_sets_are_never_empty_and_cover_cells(self):
        rng = np.random.default_rng(2)
        P, y, E = toy(6000, rng)
        obj = np.arange(4000)
        t = A.fit_table(P[:4000], y[:4000], E[:4000], obj, np.ones(4000, bool))
        ac = A.AtypicalConformal(t, list("abcdef"))
        s = ac.membership(P[4000:], E[4000:], P[4000:].argmax(1))
        self.assertTrue(s.any(1).all())
        cov = s[np.arange(2000), y[4000:]]
        g = ac.cells(E[4000:])
        for k in range(5):           # every energy quintile is covered near the nominal level
            self.assertGreater(cov[g == k].mean(), .85)

    def test_small_cells_fall_back_to_the_class_threshold(self):
        rng = np.random.default_rng(3)
        P, y, E = toy(3000, rng)
        y[:] = 0; y[:15] = 5                                         # class 5 has 15 objects over five cells
        t = A.fit_table(P, y, E, np.arange(3000), np.ones(3000, bool))
        self.assertTrue(all(v == t["q_class"][5] for v in t["q"][5]))

    @unittest.skipUnless((MODELS / A.FILENAME).exists(), "atypicality table not exported")
    def test_shipped_classifier_modes(self):
        import pandas as pd
        import flare
        clf = flare.load_classifier("bts6")
        X = pd.DataFrame([{c: np.nan for c in clf.feature_names}] * 2)
        X["n_obs_total"] = [20, 40]
        default = clf.prediction_sets(X)
        self.assertEqual(clf.set_mode(), "class")
        atyp = clf.prediction_sets(X, mode="atypicality")
        self.assertTrue(all(len(s) >= 1 for s in atyp))
        self.assertEqual(default, clf.prediction_sets(X, mode="class"))
        with self.assertRaises(ValueError):
            clf.set_mode("nonsense")
        rep = clf.conformal_report(X, mode="atypicality")[0]
        self.assertEqual(rep["set_mode"], "atypicality")
        self.assertLessEqual(rep["review_threshold"], rep["alpha"])
        self.assertEqual(rep["review_trigger"], rep["review_p"] <= rep["review_threshold"])


class ReviewTriggerTests(unittest.TestCase):
    def test_trigger_falls_back_to_the_deployed_rule(self):
        from flare import skyportal as S
        legacy = dict(argmax_excluded_from_set=True, credibility=.5)
        self.assertTrue(S.review_trigger(legacy))
        self.assertFalse(S.review_trigger(dict(argmax_excluded_from_set=False, credibility=.5)))
        self.assertFalse(S.review_trigger(dict(legacy, review_trigger=False)))
        self.assertTrue(S.review_trigger(dict(argmax_excluded_from_set=False, credibility=.5, review_trigger=True)))

    def test_threshold_never_exceeds_alpha(self):
        from flare.hierarchical import HierarchicalFlare
        h = HierarchicalFlare.__new__(HierarchicalFlare); h.alpha = .1
        h.review_p_threshold = None; self.assertEqual(h.review_threshold(), .1)
        h.review_p_threshold = .03; self.assertEqual(h.review_threshold(), .03)
        h.review_p_threshold = .16; self.assertEqual(h.review_threshold(), .1)


if __name__ == "__main__":
    unittest.main()
