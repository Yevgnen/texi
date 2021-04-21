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
        with self.assertRaises(ValueError) as e:
            create_span_mask(starts, ends, length)
            self.assertEqual(
                e.msg, "`start` and `end` should have same lengths: 1 != 2"
            )
