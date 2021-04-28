# -*- coding: utf-8 -*-

import unittest

import torch

from texi.pytorch.masking import create_span_mask


class TestFunction(unittest.TestCase):
    def test_create_span_masks(self):
        starts = [1, 3, 4]
        ends = [4, 6, 8]
        length = 10
        mask = create_span_mask(starts, ends, length)
        output = torch.tensor(
            [
                [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
            ],
            dtype=torch.int64,
        )
        self.assertTrue((mask == output).all())

    def test_create_span_masks_pass_tensor(self):
        starts = torch.tensor([1, 3, 4])
        ends = torch.tensor([4, 6, 8])
        length = 10
        mask = create_span_mask(starts, ends, length)
        output = torch.tensor(
            [
                [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
            ],
            dtype=torch.int64,
        )
        self.assertTrue((mask == output).all())

        starts = torch.tensor([[1, 3, 4]])
        ends = torch.tensor([4])
        length = 10
        with self.assertRaises(ValueError) as ctx:
            create_span_mask(starts, ends, length)
        self.assertEqual(
            str(ctx.exception), "`starts` must be 1d if passed as tensor, got ndim == 2"
        )

        starts = torch.tensor([1])
        ends = torch.tensor([[4, 6, 8]])
        length = 10
        with self.assertRaises(ValueError) as ctx:
            create_span_mask(starts, ends, length)
        self.assertEqual(
            str(ctx.exception), "`ends` must be 1d if passed as tensor, got ndim == 2"
        )

    def test_create_span_masks_empty(self):
        starts = []
        ends = []
        length = 10
        mask = create_span_mask(starts, ends, length)
        output = torch.zeros((0, 10), dtype=torch.int64)
        self.assertTrue((mask == output).all())

    def test_create_span_masks_inconsistent_lengths(self):
        starts = [1]
        ends = [2, 3]
        length = 10
        with self.assertRaises(ValueError) as ctx:
            create_span_mask(starts, ends, length)
        self.assertEqual(
            str(ctx.exception), "`start` and `end` should have same lengths: 1 != 2"
        )

        starts = torch.tensor([1])
        ends = torch.tensor([2, 3])
        length = 10
        with self.assertRaises(ValueError) as ctx:
            create_span_mask(starts, ends, length)
        self.assertEqual(
            str(ctx.exception), "`start` and `end` should have same lengths: 1 != 2"
        )


if __name__ == "__main__":
    unittest.main()
