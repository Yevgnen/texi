# -*- coding: utf-8 -*-

import unittest

import torch

from texi.pytorch.metrics.re_metrics import ReMetrics


class TestReMetrics(unittest.TestCase):
    def setUp(self):
        labels = ["born in", "loves", "NEGATIVE_RELATION"]
        relation_index2label = dict(zip(range(len(labels)), labels))
        negative_relation_index = 2
        relation_filter_threshold = 0.4

        device = torch.device("cpu")
        metric = ReMetrics(
            relation_index2label,
            negative_relation_index,
            relation_filter_threshold,
            device=device,
        )

        self.relation_index2label = relation_index2label
        self.negative_relation_index = negative_relation_index
        self.device = device
        self.metric = metric
        self.output = (
            # y_pred
            {
                "label": torch.tensor(
                    [
                        [
                            [0.9, 0.1, 0.2],
                            [0.1, 0.3, 0.8],
                            [0.2, 0.3, 0.7],
                        ],  # FP, #, #
                        [
                            [0.1, 1.0, 0.1],
                            [0.1, 0.99, 0.2],
                            [0.1, 0.5, 0.1],
                        ],  # TP, FP, FP
                    ]
                ),
                "pair": torch.tensor(
                    [
                        [[0, 2], [1, 3], [2, 3]],
                        [[0, 1], [2, 4], [5, 6]],
                    ]
                ),
                "mask": torch.tensor(
                    [
                        [1, 1, 0],
                        [1, 1, 0],
                    ]
                ),
                "entity_span": torch.tensor(
                    [
                        [
                            [1, 2],
                            [1, 9],
                            [2, 3],
                            [3, 4],
                            [4, 6],
                            [8, 9],
                            [0, 4],
                        ],
                        [
                            [1, 2],
                            [1, 1],
                            [2, 3],
                            [3, 4],
                            [4, 6],
                            [8, 9],
                            [0, 4],
                        ],
                    ]
                ),
                "entity_label": torch.tensor(
                    [
                        [0, 1, 2, 3, 4, 5, 6],
                        [0, 1, 2, 3, 4, 5, 6],
                    ]
                ),
            },
            # y
            {
                "label": torch.tensor(
                    [
                        [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
                        [[0, 1, 0], [0, 0, 1], [0, 0, 1]],
                    ]
                ),
                "pair": torch.tensor(
                    [
                        [[0, 1], [1, 2], [2, 3]],
                        [[0, 1], [2, 4], [5, 6]],
                    ]
                ),
                "mask": torch.tensor(
                    [
                        [1, 1, 0],  # FN, FN, #
                        [1, 0, 0],  # TP, #, #
                    ]
                ),
                "entity_span": torch.tensor(
                    [
                        [
                            [1, 2],
                            [1, 1],
                            [2, 3],
                            [3, 4],
                            [4, 6],
                            [8, 9],
                            [0, 4],
                        ],
                        [
                            [1, 2],
                            [1, 1],
                            [2, 3],
                            [3, 4],
                            [4, 6],
                            [8, 9],
                            [0, 4],
                        ],
                    ]
                ),
                "entity_label": torch.tensor(
                    [
                        [0, 1, 2, 3, 4, 5, 6],
                        [0, 1, 2, 3, 4, 5, 6],
                    ]
                ),
            },
        )

    def test_init(self):
        self.assertTrue(
            torch.all(self.metric.tpfpfn == torch.zeros((3,), device=self.device))
        )
        self.assertTrue(
            torch.all(
                self.metric.typed_tpfpfn
                == torch.zeros((len(self.relation_index2label), 3), device=self.device)
            )
        )

    def test_reset(self):
        self.metric.reset()

        self.assertTrue(
            torch.all(self.metric.tpfpfn == torch.zeros((3,), device=self.device))
        )
        self.assertTrue(
            torch.all(
                self.metric.typed_tpfpfn
                == torch.zeros((len(self.relation_index2label), 3), device=self.device)
            )
        )

    def test_update(self):
        self.metric.update(self.output)
        self.assertTrue(
            torch.all(
                self.metric.tpfpfn == torch.tensor([1, 2, 2]),
            )
        )
        self.assertTrue(
            torch.all(
                self.metric.typed_tpfpfn
                == torch.tensor(
                    [
                        [0, 1, 1],
                        [1, 1, 1],
                        [0, 0, 0],
                    ],
                )
            )
        )

    def test_compute(self):
        self.metric.update(self.output)
        metrics = self.metric.compute()
        precision = 1 / (1 + 2)
        recall = 1 / (1 + 2)
        f1 = 2 * precision * recall / (precision + recall)
        self.assertAlmostEqual(metrics["micro.precision"], precision)
        self.assertAlmostEqual(metrics["micro.recall"], recall)
        self.assertAlmostEqual(metrics["micro.f1"], f1)

        entities = ["born in", "loves"]
        precisions = [0, 1 / (1 + 1), 0]
        recalls = [0, 1 / (1 + 1), 0]
        f1s = [
            0,
            2 * 1 / (1 + 1) * 1 / (1 + 1) / (1 / (1 + 1) + 1 / (1 + 1)),
            0,
        ]
        for entity, p, r, f1 in zip(entities, precisions, recalls, f1s):
            self.assertAlmostEqual(metrics[f"{entity}.precision"], p)
            self.assertAlmostEqual(metrics[f"{entity}.recall"], r)
            self.assertAlmostEqual(metrics[f"{entity}.f1"], f1)


if __name__ == "__main__":
    unittest.main()
