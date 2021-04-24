# -*- coding: utf-8 -*-

import unittest

from texi.apps.ner.utils import split_example


class TestFunctions(unittest.TestCase):
    def test_split_example_empty_tokens(self):
        with self.assertRaises(ValueError) as ctx:
            split_example({"tokens": []}, ".")
        self.assertEqual(
            str(ctx.exception), "`example` should at least contain one token"
        )

    def test_split_example_normal(self):
        example = {
            "tokens": [
                "Bill",
                "was",
                "born",
                "in",
                "USA",
                ".",
                "Jack",
                "Loves",
                "Mary",
                ".",
            ],
            "entities": [
                {"type": "per", "start": 0, "end": 1},
                {"type": "loc", "start": 4, "end": 5},
                {"type": "per", "start": 6, "end": 7},
                {"type": "per", "start": 8, "end": 9},
            ],
            "relations": [
                {"type": "born in", "head": 0, "tail": 1},
                {"type": "loves", "head": 2, "tail": 3},
            ],
        }

        splits = split_example(example, ".")
        self.assertEqual(len(splits), 2)
        self.assertEqual(
            splits[0],
            {
                "tokens": ["Bill", "was", "born", "in", "USA", "."],
                "entities": [
                    {"type": "per", "start": 0, "end": 1},
                    {"type": "loc", "start": 4, "end": 5},
                ],
                "relations": [{"type": "born in", "head": 0, "tail": 1}],
            },
        )
        self.assertEqual(
            splits[1],
            {
                "tokens": ["Jack", "Loves", "Mary", "."],
                "entities": [
                    {"type": "per", "start": 6, "end": 7},
                    {"type": "per", "start": 8, "end": 9},
                ],
                "relations": [{"type": "loves", "head": 2, "tail": 3}],
            },
        )

    def test_split_example_missing_last_delimiter(self):
        example = {
            "tokens": [
                "Bill",
                "was",
                "born",
                "in",
                "USA",
                ".",
                "Jack",
                "Loves",
                "Mary",
            ],
            "entities": [
                {"type": "per", "start": 0, "end": 1},
                {"type": "loc", "start": 4, "end": 5},
                {"type": "per", "start": 6, "end": 7},
                {"type": "per", "start": 8, "end": 9},
            ],
            "relations": [
                {"type": "born in", "head": 0, "tail": 1},
                {"type": "loves", "head": 2, "tail": 3},
            ],
        }
        splits = split_example(example, ".")

        self.assertEqual(len(splits), 2)
        self.assertEqual(
            splits[0],
            {
                "tokens": ["Bill", "was", "born", "in", "USA", "."],
                "entities": [
                    {"type": "per", "start": 0, "end": 1},
                    {"type": "loc", "start": 4, "end": 5},
                ],
                "relations": [{"type": "born in", "head": 0, "tail": 1}],
            },
        )
        self.assertEqual(
            splits[1],
            {
                "tokens": ["Jack", "Loves", "Mary"],
                "entities": [
                    {"type": "per", "start": 6, "end": 7},
                    {"type": "per", "start": 8, "end": 9},
                ],
                "relations": [{"type": "loves", "head": 2, "tail": 3}],
            },
        )

    def test_split_example_entity_contains_delimiters(self):
        example = {
            "tokens": ["Bill", ".", "Gates", "born", "in", "USA"],
            "entities": [
                {"type": "per", "start": 0, "end": 2},
            ],
            "relations": [],
        }

        with self.assertRaises(RuntimeError) as ctx:
            split_example(example, ".")
        self.assertEqual(
            str(ctx.exception),
            "Entity must not contains delimiters,"
            " delimiters: {'.'}, entity: {'type': 'per', 'start': 0, 'end': 2}",
        )

    def test_split_example_relation_cross_splits(self):
        example = {
            "tokens": [
                "Bill",
                "was",
                "born",
                "in",
                "USA",
                ".",
                "Jack",
                "Loves",
                "Mary",
            ],
            "entities": [
                {"type": "per", "start": 0, "end": 1},
                {"type": "loc", "start": 4, "end": 5},
                {"type": "per", "start": 6, "end": 7},
                {"type": "per", "start": 8, "end": 9},
            ],
            "relations": [
                {"type": "born in", "head": 1, "tail": 3},
            ],
        }

        with self.assertRaises(RuntimeError) as ctx:
            split_example(example, ".")
        self.assertEqual(
            str(ctx.exception),
            "Relation must not across delimiters,"
            " delimiters: {'.'}, relation: {'type': 'born in', 'head': 1, 'tail': 3}",
        )

    def test_split_example_empty_entities_and_relations(self):
        example = {
            "tokens": [
                "Bill",
                "was",
                "born",
                "in",
                "USA",
                ".",
                "Jack",
                "Loves",
                "Mary",
            ],
            "entities": [],
            "relations": [],
        }

        splits = split_example(example, ".")
        self.assertEqual(len(splits), 2)
        self.assertEqual(
            splits[0],
            {
                "tokens": ["Bill", "was", "born", "in", "USA", "."],
                "entities": [],
                "relations": [],
            },
        )
        self.assertEqual(
            splits[1],
            {
                "tokens": ["Jack", "Loves", "Mary"],
                "entities": [],
                "relations": [],
            },
        )

    def test_split_example_invalid_relations_are_drop(self):
        example = {
            "tokens": [
                "Bill",
                "was",
                "born",
                "in",
                "USA",
                ".",
                "Jack",
                "Loves",
                "Mary",
            ],
            "entities": [],
            "relations": [{"type": "born in", "head": 1, "tail": 3}],
        }

        splits = split_example(example, ".")
        self.assertEqual(len(splits), 2)
        self.assertEqual(
            splits[0],
            {
                "tokens": ["Bill", "was", "born", "in", "USA", "."],
                "entities": [],
                "relations": [],
            },
        )
        self.assertEqual(
            splits[1],
            {
                "tokens": ["Jack", "Loves", "Mary"],
                "entities": [],
                "relations": [],
            },
        )


if __name__ == "__main__":
    unittest.main()
