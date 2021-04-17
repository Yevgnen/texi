# -*- coding: utf-8 -*-

import unittest

import torch

from texi.pytorch.transformers.spert import SpERTSampler


def random_entity_mask(num_entities, max_length=20):
    start = torch.randint(0, max_length, size=(num_entities,))
    end = torch.randint(0, max_length, size=(num_entities,))

    start, end = torch.min(start, end), torch.max(start, end)
    mask = torch.arange(max_length).view(max_length, -1)
    mask = (mask >= start) & (mask < end)
    mask = mask.transpose(0, 1)

    return mask


class TestSpERTSampler(unittest.TestCase):
    def setUp(self):
        self.example = {
            "text": ["Bill", "was", "born", "in", "USA", "."],
            "entities": [
                {"type": "per", "start": 0, "end": 1},
                {"type": "prep", "start": 3, "end": 4},
                {"type": "loc", "start": 4, "end": 5},
            ],
            "relations": [
                {
                    "type": "born in",
                    "arg1": {
                        "type": "per",
                        "start": 0,
                        "end": 1,
                    },
                    "arg2": {
                        "type": "loc",
                        "start": 4,
                        "end": 5,
                    },
                }
            ],
        }
        self.num_negative_entities = 10
        self.num_negative_relations = 10
        self.max_entity_length = 7
        self.negative_entity_type = "NOT_ENTITY"
        self.negative_relation_type = "NO_RELATION"

        self.sampler = SpERTSampler(
            self.num_negative_entities,
            self.num_negative_relations,
            self.max_entity_length,
            negative_entity_type=self.negative_entity_type,
            negative_relation_type=self.negative_relation_type,
        )

    def test_sample_negative_entities_few_entities(self):
        example = {
            "text": "abc",
            "entities": [],
        }
        sampler = SpERTSampler(
            10,
            self.num_negative_relations,
            self.max_entity_length,
            negative_entity_type=self.negative_entity_type,
        )
        negatives = sampler.sample_negative_entities(example)

        self.assertEqual(len(negatives), 6)

    def test_sample_negative_entities(self):
        negatives = self.sampler.sample_negative_entities(self.example)

        self.assertEqual(len(negatives), self.num_negative_entities)
        for negative in negatives:
            self.assertGreaterEqual(len(negative["mention"]), 1)
            self.assertLessEqual(len(negative["mention"]), self.max_entity_length)
            self.assertNotEqual(negative["mention"], ["Bill"])
            self.assertNotEqual(negative["mention"], ["USA"])
            self.assertEqual(
                self.example["text"][negative["start"] : negative["end"]],
                negative["mention"],
            )
            self.assertEqual(negative["type"], self.negative_entity_type)
            self.assertIsInstance(negative["start"], int)
            self.assertIsInstance(negative["end"], int)

    def test_sample_negative_relation(self):
        negatives = self.sampler.sample_negative_relations(self.example)

        self.assertEqual(len(negatives), 5)
        self.assertIn(
            {
                "type": self.negative_relation_type,
                "arg1": {"type": "per", "start": 0, "end": 1},
                "arg2": {"type": "prep", "start": 3, "end": 4},
            },
            negatives,
        )
        self.assertIn(
            {
                "type": self.negative_relation_type,
                "arg1": {"type": "prep", "start": 3, "end": 4},
                "arg2": {"type": "per", "start": 0, "end": 1},
            },
            negatives,
        )
        self.assertIn(
            {
                "type": self.negative_relation_type,
                "arg1": {"type": "loc", "start": 4, "end": 5},
                "arg2": {"type": "prep", "start": 3, "end": 4},
            },
            negatives,
        )
        self.assertIn(
            {
                "type": self.negative_relation_type,
                "arg1": {"type": "prep", "start": 3, "end": 4},
                "arg2": {"type": "loc", "start": 4, "end": 5},
            },
            negatives,
        )
        self.assertIn(
            {
                "type": self.negative_relation_type,
                "arg1": {"type": "loc", "start": 4, "end": 5},
                "arg2": {"type": "per", "start": 0, "end": 1},
            },
            negatives,
        )


if __name__ == "__main__":
    unittest.main()
