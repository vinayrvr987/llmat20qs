import math
import unittest

try:
    import numpy as np
    import pandas as pd
    from src.analysis import evaluation as ev
except ImportError:  # analysis dependencies are optional in the GPU environment
    ev = None


@unittest.skipIf(ev is None, "analysis dependencies (numpy, pandas) are not installed")
class AnswerKeyTests(unittest.TestCase):
    animal = {"id": "animal-pangolin", "category": "animal", "e1": True}
    place = {"id": "place-paris", "category": "place", "e1": True}
    person = {"id": "person-ada-lovelace", "category": "person", "e1": True}

    def test_category_questions_are_keyed_from_the_target(self):
        self.assertEqual(ev.keyed_answer("Is it an animal?", self.animal), "Yes")
        self.assertEqual(ev.keyed_answer("Is your concept a person?", self.animal), "No")
        self.assertEqual(ev.keyed_answer("is the secret concept a place", self.place), "Yes")

    def test_taxon_questions_use_question_bank_labels(self):
        self.assertEqual(ev.keyed_answer("Is it a mammal?", self.animal), "Yes")
        self.assertEqual(ev.keyed_answer("Is it a bird?", self.animal), "No")
        self.assertIsNone(ev.keyed_answer("Is it a mammal?", self.person))

    def test_ambiguous_or_open_questions_are_not_keyed(self):
        self.assertIsNone(ev.keyed_answer("Is it an inanimate object?", self.place))
        self.assertIsNone(ev.keyed_answer("Is it a living thing?", self.person))
        self.assertIsNone(ev.keyed_answer("Is it bigger than a breadbox?", self.animal))
        self.assertEqual(ev.keyed_answer("Is it an object?", self.person), "No")


@unittest.skipIf(ev is None, "analysis dependencies (numpy, pandas) are not installed")
class StatisticsTests(unittest.TestCase):
    def test_holm_is_monotone_and_capped(self):
        self.assertEqual(ev.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        self.assertEqual(ev.holm([0.9, 0.8]), [1.0, 1.0])

    def test_sign_flip_detects_consistent_differences_only(self):
        self.assertLess(ev.sign_flip_p([1.0] * 20, seed=1, permutations=2000), 0.01)
        self.assertEqual(ev.sign_flip_p([0.0] * 20, seed=1), 1.0)
        self.assertGreater(ev.sign_flip_p([1, -1] * 10, seed=1, permutations=2000), 0.5)

    def test_information_theory_references(self):
        self.assertEqual(ev.ulam_questions(128, 0), 7)
        self.assertEqual(ev.ulam_questions(128, 1), 11)
        self.assertEqual(ev.ulam_questions(128, 2), 14)
        self.assertAlmostEqual(ev.noisy_questions(128, 0.05) - 7, 2.81, places=2)
        self.assertEqual(ev.binary_entropy(0.5), 1.0)

    def test_wilson_interval_contains_rate(self):
        low, high = ev.wilson(3, 10)
        self.assertLess(low, 0.3)
        self.assertGreater(high, 0.3)

    def test_clustered_ols_recovers_coefficients(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=400)
        design = pd.DataFrame({"intercept": 1.0, "x": x})
        y = 2.0 + 0.5 * x + rng.normal(scale=0.1, size=400)
        table = ev.ols_clustered(design, y, np.repeat(np.arange(40), 10))
        self.assertAlmostEqual(table.loc["x", "estimate"], 0.5, places=1)
        self.assertTrue(math.isfinite(table.loc["x", "se"]))


if __name__ == "__main__":
    unittest.main()
