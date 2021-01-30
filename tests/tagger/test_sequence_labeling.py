# -*- coding: utf-8 -*-

import unittest

from texi.tagger.sequence_labeling import IOB1, IOB2, IOBES


def _get_chunks():
    inputs = {
        "tokens": [
            "Bill",
            "works",
            "for",
            "Bank",
            "of",
            "America",
            "and",
            "takes",
            "the",
            "Boston",
            "Philadelphia",
            "train.",
        ],
        "chunks": [
            {"token": ["Bill"], "tag": "PER", "start": 0, "end": 1},
            {
                "token": ["Bank", "of", "America"],
                "tag": "ORG",
                "start": 3,
                "end": 6,
            },
            {"token": ["Boston"], "tag": "LOC", "start": 9, "end": 10},
            {"token": ["Philadelphia"], "tag": "LOC", "start": 10, "end": 11},
        ],
    }

    return inputs


class TestIOB1(unittest.TestCase):
    def setUp(self):
        self.tagger = IOB1()

    def test_encode(self):
        inputs = _get_chunks()
        outputs = self.tagger.encode(inputs)
        expected = {
            "tokens": inputs["tokens"],
            "tags": [
                "I-PER",
                "O",
                "O",
                "I-ORG",
                "I-ORG",
                "I-ORG",
                "O",
                "O",
                "O",
                "I-LOC",
                "B-LOC",
                "O",
            ],
        }

        self.assertDictEqual(expected, outputs)

    def test_decode(self):
        expected = _get_chunks()
        inputs = {
            "tokens": expected["tokens"],
            "tags": [
                "I-PER",
                "O",
                "O",
                "I-ORG",
                "I-ORG",
                "I-ORG",
                "O",
                "O",
                "O",
                "I-LOC",
                "B-LOC",
                "O",
            ],
        }
        outputs = self.tagger.decode(inputs)

        self.assertDictEqual(expected, outputs)


class TestIOB2(unittest.TestCase):
    def setUp(self):
        self.tagger = IOB2()

    def test_encode(self):
        inputs = _get_chunks()
        outputs = self.tagger.encode(inputs)
        expected = {
            "tokens": inputs["tokens"],
            "tags": [
                "B-PER",
                "O",
                "O",
                "B-ORG",
                "I-ORG",
                "I-ORG",
                "O",
                "O",
                "O",
                "B-LOC",
                "B-LOC",
                "O",
            ],
        }
        self.assertDictEqual(expected, outputs)

    def test_decode(self):
        expected = _get_chunks()
        inputs = {
            "tokens": expected["tokens"],
            "tags": [
                "B-PER",
                "O",
                "O",
                "B-ORG",
                "I-ORG",
                "I-ORG",
                "O",
                "O",
                "O",
                "B-LOC",
                "B-LOC",
                "O",
            ],
        }
        outputs = self.tagger.decode(inputs)

        self.assertDictEqual(expected, outputs)


class TestIOBES(unittest.TestCase):
    def setUp(self):
        self.tagger = IOBES()

    def test_encode(self):
        inputs = _get_chunks()
        outputs = self.tagger.encode(inputs)
        expected = {
            "tokens": inputs["tokens"],
            "tags": [
                "S-PER",
                "O",
                "O",
                "B-ORG",
                "I-ORG",
                "E-ORG",
                "O",
                "O",
                "O",
                "S-LOC",
                "S-LOC",
                "O",
            ],
        }

        self.assertDictEqual(expected, outputs)

    def test_decode(self):
        expected = _get_chunks()
        inputs = {
            "tokens": expected["tokens"],
            "tags": [
                "S-PER",
                "O",
                "O",
                "B-ORG",
                "I-ORG",
                "E-ORG",
                "O",
                "O",
                "O",
                "S-LOC",
                "S-LOC",
                "O",
            ],
        }
        outputs = self.tagger.decode(inputs)

        self.assertDictEqual(expected, outputs)


if __name__ == "__main__":
    unittest.main()
