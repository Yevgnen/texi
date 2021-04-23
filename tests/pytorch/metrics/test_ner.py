# -*- coding: utf-8 -*-

import unittest

import torch

from texi.preprocessing import LabelEncoder
from texi.pytorch.metrics.ner import NerMetrics


class TestNerMetrics(unittest.TestCase):
    def setUp(self):
        label_encoder = LabelEncoder(["per", "loc", "NON_ENTITY"])
        negative_index = label_encoder.encode_label("NON_ENTITY")
        device = torch.device("cpu")
        metric = NerMetrics(label_encoder, negative_index, device=device)

        self.label_encoder = label_encoder
        self.negative_index = negative_index
        self.device = device
        self.metric = metric
        self.output = (
            # y_pred
            {
                "label": torch.tensor(
                    [
                        [1, 2, 2],  # T, #, #
                        [1, 1, 1],  # T, FP, FP
                    ]
                ),
                "span": torch.tensor(
                    [
                        [[0, 1], [1, 3], [2, 3]],
                        [[0, 1], [2, 4], [5, 6]],
                    ]
                ),
                "mask": torch.tensor(
                    [
                        [1, 1, 0],
                        [1, 1, 0],
                    ]
                ),
            },
            # y
            {
                "label": torch.tensor(
                    [
                        [1, 0, 2],
                        [1, 2, 2],
                    ]
                ),
                "span": torch.tensor(
                    [
                        [[0, 1], [1, 2], [2, 3]],
                        [[0, 1], [2, 4], [5, 6]],
                    ]
                ),
                "mask": torch.tensor(
                    [
                        [1, 1, 0],  # T, FN, #
                        [1, 0, 0],  # T, #, #
                    ]
                ),
            },
        )

    def test_init(self):
        self.assertTrue(
            torch.all(self.metric.entity_stat == torch.zeros((3,), device=self.device))
        )
        self.assertTrue(
            torch.all(
                self.metric.typed_entity_stat
                == torch.zeros((len(self.label_encoder), 3), device=self.device)
            )
        )

    def test_reset(self):
        self.metric.reset()

        self.assertTrue(
            torch.all(self.metric.entity_stat == torch.zeros((3,), device=self.device))
        )
        self.assertTrue(
            torch.all(
                self.metric.typed_entity_stat
                == torch.zeros((len(self.label_encoder), 3), device=self.device)
            )
        )

    def test_update(self):
        self.metric.update(self.output)
        self.assertTrue(
            torch.all(
                self.metric.entity_stat == torch.tensor([2, 2, 1]),
            )
        )
        self.assertTrue(
            torch.all(
                self.metric.typed_entity_stat
                == torch.tensor(
                    [
                        [0, 0, 1],
                        [2, 2, 0],
                        [0, 0, 0],
                    ],
                )
            )
        )

    def test_compute(self):
        self.metric.update(self.output)
        metrics = self.metric.compute()
        precision = 2 / (2 + 2)
        recall = 2 / (2 + 1)
        f1 = 2 * precision * recall / (precision + recall)
        self.assertAlmostEqual(metrics["all"]["precision"], precision)
        self.assertAlmostEqual(metrics["all"]["recall"], recall)
        self.assertAlmostEqual(metrics["all"]["f1"], f1)

        relations = ["per", "loc", "NON_ENTITY"]
        precisions = [0, 2 / (2 + 2), 0]
        recalls = [0, 2 / (2 + 0), 0]
        f1s = [
            0,
            2 * 2 / (2 + 2) * 2 / (2 + 0) / (2 / (2 + 2) + 2 / (2 + 0)),
            0,
        ]
        for relation, p, r, f1 in zip(relations, precisions, recalls, f1s):
            self.assertAlmostEqual(metrics[relation]["precision"], p)
            self.assertAlmostEqual(metrics[relation]["recall"], r)
            self.assertAlmostEqual(metrics[relation]["f1"], f1)


if __name__ == "__main__":
    unittest.main()
