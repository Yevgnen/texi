# -*- coding: utf-8 -*-

import unittest

import torch

from texi.pytorch.utils import split_tensors


class TestSplitTensors(unittest.TestCase):
    def _valid_output(self, output, expected):
        self.assertIsInstance(output, list)
        for x in output:
            self.assertIsInstance(x, tuple)
        for x, y in zip(output, expected):
            for xi, yi in zip(x, y):
                self.assertTrue(torch.all(xi == yi))

    def test_split_tensors(self):
        tensors = [
            torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
            torch.tensor([[0, 0, 1, 1], [2, 2, 3, 3]]),
        ]
        output = split_tensors(tensors, 1, dim=0)
        expected = [
            (torch.tensor([[1, 2, 3, 4]]), torch.tensor([[0, 0, 1, 1]])),
            (torch.tensor([[5, 6, 7, 8]]), torch.tensor([[2, 2, 3, 3]])),
        ]
        self._valid_output(output, expected)

        output = split_tensors(tensors, 2, dim=1)
        expected = [
            (torch.tensor([[1, 2], [5, 6]]), torch.tensor([[0, 0], [2, 2]])),
            (torch.tensor([[3, 4], [7, 8]]), torch.tensor([[1, 1], [3, 3]])),
        ]
        self._valid_output(output, expected)

    def test_split_tensors_empty_tensors(self):
        tensors = []
        with self.assertRaises(ValueError) as ctx:
            split_tensors(tensors, 1, dim=1)
        self.assertEqual(
            str(ctx.exception),
            "At least one tensor must be given.",
        )

    def test_split_tensors_mismatched_dimension(self):
        tensors = [
            torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
            torch.tensor([[0, 0, 1], [2, 2, 3]]),
        ]
        with self.assertRaises(ValueError) as ctx:
            split_tensors(tensors, 1, dim=1)
        self.assertEqual(
            str(ctx.exception),
            "Size of dimension `dim` = 1 "
            "must be same for all input tensors, "
            "got: [(2, 4), (2, 3)].",
        )


if __name__ == "__main__":
    unittest.main()
