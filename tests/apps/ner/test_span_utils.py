# -*- coding: utf-8 -*-

import unittest

from texi.apps.ner.span_utils import SpanNerNegativeSampler


class TestSpanNerNegativeSampler(unittest.TestCase):
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
        self.negative_entity_type = "NEGATIVE_ENTITY"
        self.negative_relation_type = "NEGATIVE_RELATION"

        self.sampler = SpanNerNegativeSampler(
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
        sampler = SpanNerNegativeSampler(
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
        sampler = SpanNerNegativeSampler(
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


if __name__ == "__main__":
    unittest.main()
