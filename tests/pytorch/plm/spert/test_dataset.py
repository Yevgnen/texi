# -*- coding: utf-8 -*-

import unittest
from unittest.mock import Mock

import torch
from transformers import BertTokenizer

from texi.preprocessing import LabelEncoder
from texi.pytorch.models.spert import SpERTDataset
from texi.pytorch.plm.utils import plm_path


class TestSpERTDataset(unittest.TestCase):
    def setUp(self):
        self.example = {
            "tokens": ["BillGates", "was", "born", "in", "America", "."],
            "entities": [
                {"type": "per", "start": 0, "end": 1},
                {"type": "prep", "start": 3, "end": 4},
                {"type": "loc", "start": 4, "end": 5},
                {"type": "NEGATIVE_ENTITY", "start": 0, "end": 3},
            ],
            "relations": [
                {"type": "born in", "head": 0, "tail": 2},
                {"type": "NEGATIVE_RELATION", "head": 1, "tail": 2},
                {"type": "NEGATIVE_RELATION", "head": 2, "tail": 0},
            ],
        }

    def test_encode_example(self):
        entity_label_encoder = LabelEncoder(["per", "prep", "loc", "NEGATIVE_ENTITY"])
        relation_label_encoder = LabelEncoder(["born in", "NEGATIVE_RELATION"])

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
