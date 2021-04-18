# -*- coding: utf-8 -*-

import unittest
from unittest.mock import Mock

import torch
from transformers import BertTokenizerFast

from texi.preprocessing import LabelEncoder
from texi.pytorch.transformers.spert import SpERTDataset, SpERTSampler, create_span_mask


def random_entity_mask(num_entities, max_length=20):
    start = torch.randint(0, max_length, size=(num_entities,))
    end = torch.randint(0, max_length, size=(num_entities,))

    start, end = torch.min(start, end), torch.max(start, end)
    mask = torch.arange(max_length).view(max_length, -1)
    mask = (mask >= start) & (mask < end)
    mask = mask.transpose(0, 1)

    return mask


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
                    "arg1": {"type": "per", "start": 0, "end": 1},
                    "arg2": {"type": "loc", "start": 4, "end": 5},
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
            "text": ["I", "hate", "you"],
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

    def test_sample_negative_relation_no_entities(self):
        example = {
            "text": ["I", "hate", "you"],
            "entities": [],
        }
        sampler = SpERTSampler(
            self.num_negative_entities,
            10,
            self.max_entity_length,
            negative_entity_type=self.negative_entity_type,
        )
        negatives = sampler.sample_negative_relations(example)
        self.assertEqual(len(negatives), 0)

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


class TestSpERTDataset(unittest.TestCase):
    def setUp(self):
        self.example = {
            "text": ["BillGates", "was", "born", "in", "America", "."],
            "entities": [
                {"type": "per", "start": 0, "end": 1},
                {"type": "prep", "start": 3, "end": 4},
                {"type": "loc", "start": 4, "end": 5},
            ],
            "relations": [
                {
                    "type": "born in",
                    "arg1": {"type": "per", "start": 0, "end": 1},
                    "arg2": {"type": "loc", "start": 4, "end": 5},
                }
            ],
        }
        self.entities = [
            {"type": "per", "start": 0, "end": 1},
            {"type": "prep", "start": 3, "end": 4},
            {"type": "loc", "start": 4, "end": 5},
            {"type": "NOT_ENTITY", "start": 0, "end": 3},
        ]
        self.relations = [
            {
                "type": "born in",
                "arg1": {"type": "per", "start": 0, "end": 1},
                "arg2": {"type": "loc", "start": 4, "end": 5},
            },
            {
                "type": "NO_RELATION",
                "arg1": {"type": "prep", "start": 3, "end": 4},
                "arg2": {"type": "loc", "start": 4, "end": 5},
            },
            {
                "type": "NO_RELATION",
                "arg1": {"type": "loc", "start": 4, "end": 5},
                "arg2": {"type": "per", "start": 0, "end": 1},
            },
        ]

    def test_encode_example(self):
        entity_label_encoder = LabelEncoder(["per", "prep", "loc", "NOT_ENTITY"])
        relation_label_encoder = LabelEncoder(["born in", "NO_RELATION"])

        dataset = SpERTDataset(
            [self.example],
            Mock(),
            entity_label_encoder,
            relation_label_encoder,
            tokenizer=BertTokenizerFast.from_pretrained("bert-base-uncased"),
        )
        output = dataset.encode_example(self.example, self.entities, self.relations)

        self.assertEqual(len(output["entities"]), len(self.entities))
        self.assertEqual(
            set(output["output"].keys()),
            {"input_ids", "token_type_ids", "attention_mask"},
        )
        self.assertEqual(
            output["entities"][0],
            {"type": 0, "start": 1, "end": 4},
        )
        self.assertEqual(
            output["entities"][1],
            {"type": 1, "start": 6, "end": 7},
        )
        self.assertEqual(
            output["entities"][2],
            {"type": 2, "start": 7, "end": 8},
        )
        self.assertEqual(
            output["entities"][3],
            {"type": 3, "start": 1, "end": 6},
        )

        self.assertEqual(len(output["relations"]), len(self.relations))
        self.assertEqual(output["relations"][0], {"type": 0, "head": 0, "tail": 2})
        self.assertEqual(output["relations"][1], {"type": 1, "head": 1, "tail": 2})
        self.assertEqual(output["relations"][2], {"type": 1, "head": 2, "tail": 0})


if __name__ == "__main__":
    unittest.main()
