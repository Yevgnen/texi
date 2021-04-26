# -*- coding: utf-8 -*-

import unittest

from texi.metrics import prf1, tpfpfn


class TestFunctions(unittest.TestCase):
    def test_prf1(self):
        tp, fp, fn = 1, 2, 3
        output = prf1(tp, fp, fn)
        self.assertAlmostEqual(output["precision"], 1 / 3)
        self.assertAlmostEqual(output["recall"], 1 / 4)
        self.assertAlmostEqual(output["f1"], 1 / 3 * 1 / 4 * 2 / (1 / 3 + 1 / 4))

    def test_prf1_zero_precision(self):
        tp, fp, fn = 0, 2, 0
        output = prf1(tp, fp, fn)
        self.assertAlmostEqual(output["precision"], 0)

    def test_prf1_zero_recall(self):
        tp, fp, fn = 0, 0, 3
        output = prf1(tp, fp, fn)
        self.assertAlmostEqual(output["recall"], 0)

    def test_prf1_zero_f1(self):
        tp, fp, fn = 0, 0, 0
        output = prf1(tp, fp, fn)
        self.assertAlmostEqual(output["precision"], 0)
        self.assertAlmostEqual(output["recall"], 0)
        self.assertAlmostEqual(output["f1"], 0)

    def tpfpfn_return_index(self):
        y = [1, 2, 3, 4, 5]
        y_pred = [3, 4, 5, 6, 7]
        output = tpfpfn(y, y_pred, return_index=True)
        self.assertEqual(
            output,
            {
                "tp": [2, 3, 4],
                "fp": [3, 4],
                "fn": [0, 1],
            },
        )

    def tpfpfn_return_value(self):
        y = [1, 2, 3, 4, 5]
        y_pred = [3, 4, 5, 6, 7]
        output = tpfpfn(y, y_pred, return_index=False)
        self.assertEqual(
            output,
            {
                "tp": [3, 4, 5],
                "fp": [6, 7],
                "fn": [1, 2],
            },
        )


if __name__ == "__main__":
    unittest.main()
