# -*- coding: utf-8 -*-

import unittest

from texi.apps.ner import NerEvaluator


class TestNerEvaluator(unittest.TestCase):
    def setUp(self):
        self.types = ["per", "loc"]
        self.evaluator = NerEvaluator(self.types)
        self.targets = [
            [
                {"type": "per", "start": 0, "end": 1},
                {"type": "loc", "start": 4, "end": 5},
            ],
            [
                {"type": "per", "start": 0, "end": 1},
                {"type": "per", "start": 2, "end": 3},
            ],
            [],
        ]
        self.predictions = [
            [
                {"type": "per", "start": 0, "end": 1},
            ],
            [
                {"type": "per", "start": 0, "end": 1},
                {"type": "per", "start": 1, "end": 2},
                {"type": "per", "start": 2, "end": 3},
            ],
            [],
        ]

    def test_init(self):
        self.assertEqual(len(self.evaluator.counter), 0)

    def test_reset(self):
        self.evaluator.update(self.targets, self.predictions)
        self.evaluator.reset()
        self.assertEqual(len(self.evaluator.counter), 0)

    def test_update(self):
        self.evaluator.update(self.targets, self.predictions)
        self.assertEqual(self.evaluator.counter["tp"], 2)
        self.assertEqual(self.evaluator.counter["fp"], 1)
        self.assertEqual(self.evaluator.counter["fn"], 1)
        self.assertEqual(self.evaluator.counter["per_tp"], 2)
        self.assertEqual(self.evaluator.counter["per_fp"], 1)
        self.assertEqual(self.evaluator.counter["per_fn"], 0)
        self.assertEqual(self.evaluator.counter["loc_tp"], 0)
        self.assertEqual(self.evaluator.counter["loc_fp"], 0)
        self.assertEqual(self.evaluator.counter["loc_fn"], 1)

    def test_update_inconsistent_lengths(self):
        with self.assertRaises(ValueError) as e:
            self.evaluator.update([[]], [[], []])
            self.assertEqual(
                e.msg, "`targets` and `predictions` must have same lengths: 1 != 2"
            )

    def test_compute(self):
        self.evaluator.update(self.targets, self.predictions)
        metrics, type_metrics = self.evaluator.compute()
        self.assertEqual(len(self.evaluator.counter), 0)
        self.assertEqual(len(type_metrics), 2)
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["recall"], 2 / 3)
        self.assertAlmostEqual(metrics["f1"], 2 / 3)
        self.assertAlmostEqual(type_metrics["per"]["precision"], 2 / 3)
        self.assertAlmostEqual(type_metrics["per"]["recall"], 2 / 2)
        self.assertAlmostEqual(
            type_metrics["per"]["f1"], 2 * (2 / 3) * (2 / 2) / ((2 / 3) + (2 / 2))
        )
        self.assertAlmostEqual(type_metrics["loc"]["precision"], 0)
        self.assertAlmostEqual(type_metrics["loc"]["recall"], 0)
        self.assertAlmostEqual(type_metrics["loc"]["f1"], 0)


if __name__ == "__main__":
    unittest.main()
