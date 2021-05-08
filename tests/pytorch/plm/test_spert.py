# -*- coding: utf-8 -*-

import unittest
from unittest.mock import Mock

import torch
from transformers import BertTokenizer

from texi.preprocessing import LabelEncoder
from texi.pytorch.plm.spert import SpERTDataset, SpERTSampler
from texi.pytorch.plm.utils import plm_path


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
            "tokens": ["Bill", "was", "born", "in", "USA", "."],
            "entities": [
                {"type": "per", "start": 0, "end": 1},
                {"type": "prep", "start": 3, "end": 4},
                {"type": "loc", "start": 4, "end": 5},
            ],
            "relations": [{"type": "born in", "head": 0, "tail": 2}],
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
            "tokens": ["I", "hate", "you"],
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

        spans = {(0, 1), (3, 4), (4, 5)}
        for negative in negatives:
            self.assertNotIn((negative["start"], negative["end"]), spans)
            self.assertEqual(negative["type"], self.negative_entity_type)
            self.assertIsInstance(negative["start"], int)
            self.assertIsInstance(negative["end"], int)

    def test_sample_negative_relation_no_entities(self):
        example = {
            "tokens": ["I", "hate", "you"],
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
        for head, tail in ((2, 0), (0, 1), (1, 0), (1, 2), (2, 1)):
            self.assertIn(
                {"type": self.negative_relation_type, "head": head, "tail": tail},
                negatives,
            )


class TestSpERTDataset(unittest.TestCase):
    def setUp(self):
        self.example = {
            "tokens": ["BillGates", "was", "born", "in", "America", "."],
            "entities": [
                {"type": "per", "start": 0, "end": 1},
                {"type": "prep", "start": 3, "end": 4},
                {"type": "loc", "start": 4, "end": 5},
                {"type": "NOT_ENTITY", "start": 0, "end": 3},
            ],
            "relations": [
                {"type": "born in", "head": 0, "tail": 2},
                {"type": "NO_RELATION", "head": 1, "tail": 2},
                {"type": "NO_RELATION", "head": 2, "tail": 0},
            ],
        }

    def test_encode_example(self):
        entity_label_encoder = LabelEncoder(["per", "prep", "loc", "NOT_ENTITY"])
        relation_label_encoder = LabelEncoder(["born in", "NO_RELATION"])

        dataset = SpERTDataset(
            [self.example],
            Mock(),
            entity_label_encoder,
            relation_label_encoder,
            tokenizer=BertTokenizer.from_pretrained(plm_path("bert-base-uncased")),
        )
        output = dataset.encode_example(
            self.example["tokens"], self.example["entities"], self.example["relations"]
        )
        keys = {
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "entity_mask",
            "entity_label",
            "entity_span",
            "entity_sample_mask",
            "relation_mask",
            "relation_label",
            "relation",
            "relation_sample_mask",
        }
        self.assertTrue(set(output), keys)
        self.assertTrue(
            torch.all(
                output["entity_mask"]
                == torch.tensor(
                    [
                        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                    ]
                ),
            )
        )
        self.assertTrue(torch.all(output["entity_label"] == torch.tensor([0, 1, 2, 3])))
        self.assertTrue(
            torch.all(
                output["entity_span"] == torch.tensor([[0, 1], [3, 4], [4, 5], [0, 3]]),
            )
        )
        self.assertTrue(
            torch.all(output["entity_sample_mask"] == torch.tensor([1, 1, 1, 1]))
        )
        self.assertTrue(
            torch.all(
                output["relation_context_mask"]
                == torch.tensor(
                    [
                        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                    ]
                ),
            )
        )
        self.assertTrue(
            torch.all(
                output["relation_label"] == torch.tensor([[1, 0], [0, 1], [0, 1]])
            )
        )
        self.assertTrue(
            torch.all(output["relation"] == torch.tensor([[0, 2], [1, 2], [2, 0]]))
        )
        self.assertTrue(
            torch.all(output["relation_sample_mask"] == torch.tensor([1, 1, 1]))
        )


if __name__ == "__main__":
    unittest.main()
