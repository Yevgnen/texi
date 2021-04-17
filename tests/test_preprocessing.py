# -*- coding: utf-8 -*-

import unittest

import torch

from texi.preprocessing import LabelEncoder


class TestLabelEncoder(unittest.TestCase):
    def valid_encoder(self, encoder, vocab):
        self.assertEqual(len(encoder), len(vocab))
        self.assertEqual(encoder.label2index, dict(zip(vocab, range(len(vocab)))))
        self.assertEqual(encoder.index2label, dict(zip(range(len(vocab)), vocab)))
        for i, label in enumerate(vocab):
            self.assertEqual(encoder.encode_label(label), i)
            self.assertEqual(encoder.decode_label(i), label)

    def test_init(self):
        tokens = ["dog", "cat", "cat"]
        encoder = LabelEncoder(tokens)
        self.valid_encoder(encoder, ["dog", "cat"])

    def test_init_unknown(self):
        tokens = ["dog"]
        encoder = LabelEncoder(tokens, unknown="unk")
        self.valid_encoder(encoder, ["unk", "dog"])

    def test_add(self):
        tokens = ["dog"]
        encoder = LabelEncoder(tokens)
        encoder.add("cat")
        self.valid_encoder(encoder, ["dog", "cat"])

    def test_encode_label(self):
        tokens = ["dog"]
        encoder = LabelEncoder(tokens)
        self.assertEqual(encoder.encode_label("dog"), 0)
        self.assertEqual(
            encoder.encode_label("dog", return_tensors="pt"),
            torch.tensor(0, dtype=torch.int64),
        )
        with self.assertRaises(ValueError) as e:
            encoder.encode_label("dog", return_tensors="tensor")
            self.assertEqual(e.msg, '`return_tensors` should be "pt" or None')

    def test_decode_label(self):
        tokens = ["dog"]
        encoder = LabelEncoder(tokens)
        self.assertEqual(encoder.decode_label(0), "dog")
        self.assertEqual(
            encoder.decode_label(torch.tensor(0, dtype=torch.int64)), "dog"
        )
        with self.assertRaises(ValueError) as e:
            encoder.decode_label([0])
            self.assertEqual(
                e.msg, "`index` should be int or torch.LongTensor, got: list"
            )

        with self.assertRaises(ValueError) as e:
            encoder.decode_label(torch.tensor([0], dtype=torch.int64))
            self.assertEqual(
                e.msg, "tensor should be 0-d tensor, got: ndim == {index.ndim}"
            )

    def test_encode(self):
        tokens = ["dog", "cat"]
        encoder = LabelEncoder(tokens)
        self.assertEqual(encoder.encode(["dog", "cat"]), [0, 1])
        self.assertTrue(
            (
                encoder.encode(["dog", "cat"], return_tensors="pt")
                == torch.tensor([0, 1], dtype=torch.int64)
            ).all()
        )
        with self.assertRaises(ValueError) as e:
            encoder.encode(["dog"], return_tensors="tensor")
            self.assertEqual(e.msg, '`return_tensors` should be "pt" or None')

    def test_decode(self):
        tokens = ["dog", "cat"]
        encoder = LabelEncoder(tokens)
        self.assertEqual(encoder.decode([0, 1]), ["dog", "cat"])
        self.assertEqual(
            encoder.decode(torch.tensor([0, 1], dtype=torch.int64)),
            ["dog", "cat"],
        )

        with self.assertRaises(ValueError) as e:
            encoder.decode(torch.tensor([[0, 1]], dtype=torch.int64))
            self.assertEqual(e.msg, '`return_tensors` should be "pt" or None')

    def test_from_iterable(self):
        tokens = [["dog"], ["cat"]]
        encoder = LabelEncoder.from_iterable(tokens)
        self.valid_encoder(encoder, ["dog", "cat"])


if __name__ == "__main__":
    unittest.main()
