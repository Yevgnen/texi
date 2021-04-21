# -*- coding: utf-8 -*-

import unittest

from texi.apps.ner import NerEvaluator, ReEvaluator


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


class TestReEvaluator(unittest.TestCase):
    def setUp(self):
        self.types = ["born in"]
        self.strict_evaluator = ReEvaluator(self.types, strict=True)
        self.evaluator = ReEvaluator(self.types, strict=False)
        self.entity_targets = [
            [
                {"type": "per", "start": 0, "end": 1},
                {"type": "per", "start": 2, "end": 3},
                {"type": "loc", "start": 4, "end": 5},
            ],
        ]
        self.entity_predictions = [
            [
                {"type": "per", "start": 0, "end": 1},
                {"type": "per", "start": 2, "end": 4},
                {"type": "loc", "start": 4, "end": 5},
            ],
        ]
        self.targets = [
            [
                {"type": "born in", "head": 0, "tail": 1},
                {"type": "born in", "head": 0, "tail": 2},
            ],
        ]
        self.predictions = [
            [
                {"type": "born in", "head": 0, "tail": 1},
                {"type": "born in", "head": 0, "tail": 2},
            ],
        ]

    def test_init(self):
        self.assertEqual(len(self.evaluator.counter), 0)

    def test_reset(self):
        self.evaluator.update(self.targets, self.predictions)
        self.evaluator.reset()
        self.assertEqual(len(self.evaluator.counter), 0)

    def test_update_strict(self):
        self.strict_evaluator.update(
            self.targets, self.predictions, self.entity_targets, self.entity_predictions
        )
        self.assertEqual(self.strict_evaluator.counter["tp"], 1)
        self.assertEqual(self.strict_evaluator.counter["fp"], 1)
        self.assertEqual(self.strict_evaluator.counter["fn"], 1)
        self.assertEqual(self.strict_evaluator.counter["born in_tp"], 1)
        self.assertEqual(self.strict_evaluator.counter["born in_fp"], 1)
        self.assertEqual(self.strict_evaluator.counter["born in_fn"], 1)

    def test_update(self):
        self.evaluator.update(self.targets, self.predictions)
        self.assertEqual(self.evaluator.counter["tp"], 2)
        self.assertEqual(self.evaluator.counter["fp"], 0)
        self.assertEqual(self.evaluator.counter["fn"], 0)
        self.assertEqual(self.evaluator.counter["born in_tp"], 2)
        self.assertEqual(self.evaluator.counter["born in_fp"], 0)
        self.assertEqual(self.evaluator.counter["born in_fn"], 0)

    def test_update_inconsistent_lengths(self):
        with self.assertRaises(ValueError) as e:
            self.evaluator.update([[]], [[], []])
            self.assertEqual(
                e.msg, "`targets` and `predictions` must have same lengths: 1 != 2"
            )

        with self.assertRaises(ValueError) as e:
            self.strict_evaluator.update([[], []], [[], []], [[]], [[], []])
            self.assertEqual(
                e.msg, "`targets` and `predictions` must have same lengths: 1 != 2"
            )

    def test_update_strict_no_entity(self):
        with self.assertRaises(ValueError) as e:
            self.strict_evaluator.update([[]], [[], []])
            self.assertEqual(
                e.msg,
                (
                    "`entity_targets` and `entity_predictions` must not be None"
                    " when `self.strict` is True"
                ),
            )

    def test_compute(self):
        self.evaluator.update(self.targets, self.predictions)
        metrics, type_metrics = self.evaluator.compute()
        self.assertEqual(len(self.evaluator.counter), 0)
        self.assertEqual(len(type_metrics), 1)
        self.assertAlmostEqual(metrics["precision"], 1)
        self.assertAlmostEqual(metrics["recall"], 1)
        self.assertAlmostEqual(metrics["f1"], 1)
        self.assertAlmostEqual(type_metrics["born in"]["precision"], 1)
        self.assertAlmostEqual(type_metrics["born in"]["recall"], 1)
        self.assertAlmostEqual(type_metrics["born in"]["f1"], 1)

    def test_compute_strict(self):
        self.strict_evaluator.update(
            self.targets, self.predictions, self.entity_targets, self.entity_predictions
        )
        metrics, type_metrics = self.strict_evaluator.compute()
        self.assertEqual(len(self.strict_evaluator.counter), 0)
        self.assertEqual(len(type_metrics), 1)
        self.assertAlmostEqual(metrics["precision"], 1 / 2)
        self.assertAlmostEqual(metrics["recall"], 1 / 2)
        self.assertAlmostEqual(metrics["f1"], 1 / 2)
        self.assertAlmostEqual(type_metrics["born in"]["precision"], 1 / 2)
        self.assertAlmostEqual(type_metrics["born in"]["recall"], 1 / 2)
        self.assertAlmostEqual(type_metrics["born in"]["f1"], 1 / 2)


if __name__ == "__main__":
    unittest.main()
