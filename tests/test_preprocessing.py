# -*- coding: utf-8 -*-

import unittest

import torch

from texi.preprocessing import LabelEncoder


class TestLabelEncoder(unittest.TestCase):
    def valid_encoder(self, encoder, vocab):
        self.assertEqual(len(encoder), len(vocab))
        self.assertEqual(len(encoder.labels), len(vocab))
        self.assertEqual(encoder.num_labels, len(vocab))
        self.assertEqual(encoder.label2index, dict(zip(vocab, range(len(vocab)))))
        self.assertEqual(encoder.index2label, dict(zip(range(len(vocab)), vocab)))
        for i, label in enumerate(vocab):
            self.assertEqual(encoder.encode_label(label), i)
            self.assertEqual(encoder.decode_label(i), label)

    def test_init(self):
        labels = ["dog", "cat", "cat"]
        encoder = LabelEncoder(labels)
        self.valid_encoder(encoder, ["dog", "cat"])

    def test_init_default(self):
        labels = ["dog"]
        encoder = LabelEncoder(labels, default="unk")
        self.valid_encoder(encoder, ["unk", "dog"])

    def test_get_index(self):
        labels = ["dog", "cat"]
        encoder = LabelEncoder(labels)
        self.assertEqual(encoder.get_index("dog"), 0)
        self.assertEqual(encoder.get_index("cat"), 1)
        with self.assertRaises(KeyError) as ctx:
            encoder.get_index("pig")
        self.assertEqual(str(ctx.exception), "'pig'")

    def test_get_index_default(self):
        labels = ["dog", "cat"]
        encoder = LabelEncoder(labels, default="unk")
        self.assertEqual(encoder.get_index("dog"), 1)
        self.assertEqual(encoder.get_index("cat"), 2)
        self.assertEqual(encoder.get_index("pig"), 0)

    def test_get_label(self):
        labels = ["dog", "cat"]
        encoder = LabelEncoder(labels)
        self.assertEqual(encoder.get_label(0), "dog")
        self.assertEqual(encoder.get_label(1), "cat")
        with self.assertRaises(KeyError) as ctx:
            encoder.get_label(99)
        self.assertEqual(ctx.exception.args, (99,))

    def test_get_label_default(self):
        labels = ["dog", "cat"]
        encoder = LabelEncoder(labels, default="unk")
        self.assertEqual(encoder.get_label(1), "dog")
        self.assertEqual(encoder.get_label(2), "cat")
        self.assertEqual(encoder.get_label(0), "unk")

    def test_add(self):
        labels = ["dog"]
        encoder = LabelEncoder(labels)
        encoder.add("cat")
        self.valid_encoder(encoder, ["dog", "cat"])

    def test_encode_label(self):
        labels = ["dog"]
        encoder = LabelEncoder(labels)
        self.assertEqual(encoder.encode_label("dog"), 0)
        self.assertEqual(
            encoder.encode_label("dog", return_tensors="pt"),
            torch.tensor(0, dtype=torch.int64),
        )
        with self.assertRaises(ValueError) as ctx:
            encoder.encode_label("dog", return_tensors="tensor")
        self.assertEqual(str(ctx.exception), '`return_tensors` should be "pt" or None')

    def test_decode_label(self):
        labels = ["dog"]
        encoder = LabelEncoder(labels)
        self.assertEqual(encoder.decode_label(0), "dog")
        self.assertEqual(
            encoder.decode_label(torch.tensor(0, dtype=torch.int64)), "dog"
        )
        with self.assertRaises(ValueError) as ctx:
            encoder.decode_label([0])
        self.assertEqual(
            str(ctx.exception), "`index` should be int or torch.LongTensor, got: list"
        )

        with self.assertRaises(ValueError) as ctx:
            encoder.decode_label(torch.tensor([0], dtype=torch.int64))
        self.assertEqual(
            str(ctx.exception), "tensor should be 0-d tensor, got: ndim == 1"
        )

    def test_encode(self):
        labels = ["dog", "cat"]
        encoder = LabelEncoder(labels)
        self.assertEqual(encoder.encode(["dog", "cat"]), [0, 1])
        self.assertTrue(
            (
                encoder.encode(["dog", "cat"], return_tensors="pt")
                == torch.tensor([0, 1], dtype=torch.int64)
            ).all()
        )
        with self.assertRaises(ValueError) as ctx:
            encoder.encode(["dog"], return_tensors="tensor")
        self.assertEqual(str(ctx.exception), '`return_tensors` should be "pt" or None')

    def test_decode(self):
        labels = ["dog", "cat"]
        encoder = LabelEncoder(labels)
        self.assertEqual(encoder.decode([0, 1]), ["dog", "cat"])
        self.assertEqual(
            encoder.decode(torch.tensor([0, 1], dtype=torch.int64)),
            ["dog", "cat"],
        )

        with self.assertRaises(ValueError) as ctx:
            encoder.decode(torch.tensor([[0, 1]], dtype=torch.int64))
        self.assertEqual(
            str(ctx.exception), "tensor should be 1-d tensor, got: ndim == 2"
        )

    def test_from_iterable(self):
        labels = [["dog"], ["cat"]]
        encoder = LabelEncoder.from_iterable(labels)
        self.valid_encoder(encoder, ["dog", "cat"])


if __name__ == "__main__":
    unittest.main()
