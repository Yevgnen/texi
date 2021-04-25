# -*- coding: utf-8 -*-

import unittest

from texi.apps.ner.utils import merge_examples, split_example, texify_example


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
                    {"type": "per", "start": 0, "end": 1},
                    {"type": "per", "start": 2, "end": 3},
                ],
                "relations": [{"type": "loves", "head": 0, "tail": 1}],
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
                    {"type": "per", "start": 0, "end": 1},
                    {"type": "per", "start": 2, "end": 3},
                ],
                "relations": [{"type": "loves", "head": 0, "tail": 1}],
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

    def test_split_example_entity_contains_delimiters_ignore_error(self):
        example = {
            "tokens": [
                "Bill",
                ".",
                "Gates",
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
                {"type": "per", "start": 0, "end": 3},
                {"type": "loc", "start": 6, "end": 7},
                {"type": "per", "start": 8, "end": 9},
                {"type": "per", "start": 10, "end": 11},
            ],
            "relations": [],
        }
        splits = split_example(example, ".", ignore_errors=True)
        self.assertEqual(len(splits), 3)
        self.assertEqual(
            splits[0], {"tokens": ["Bill", "."], "entities": [], "relations": []}
        )
        self.assertEqual(
            splits[1],
            {
                "tokens": ["Gates", "was", "born", "in", "USA", "."],
                "entities": [
                    {"type": "loc", "start": 4, "end": 5},
                ],
                "relations": [],
            },
        )
        self.assertEqual(
            splits[2],
            {
                "tokens": [
                    "Jack",
                    "Loves",
                    "Mary",
                ],
                "entities": [
                    {"type": "per", "start": 0, "end": 1},
                    {"type": "per", "start": 2, "end": 3},
                ],
                "relations": [],
            },
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

    def test_split_example_relation_cross_splits_ignore_error(self):
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
                {"type": "fake", "head": 1, "tail": 0},
                {"type": "fake", "head": 2, "tail": 3},
            ],
        }
        splits = split_example(example, ".", ignore_errors=True)
        self.assertEqual(len(splits), 2)
        self.assertEqual(
            splits[0],
            {
                "tokens": [
                    "Bill",
                    "was",
                    "born",
                    "in",
                    "USA",
                    ".",
                ],
                "entities": [
                    {"type": "per", "start": 0, "end": 1},
                    {"type": "loc", "start": 4, "end": 5},
                ],
                "relations": [
                    {"type": "fake", "head": 1, "tail": 0},
                ],
            },
        )
        self.assertEqual(
            splits[1],
            {
                "tokens": [
                    "Jack",
                    "Loves",
                    "Mary",
                ],
                "entities": [
                    {"type": "per", "start": 0, "end": 1},
                    {"type": "per", "start": 2, "end": 3},
                ],
                "relations": [
                    {"type": "fake", "head": 0, "tail": 1},
                ],
            },
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

    def test_merge_examples_normal(self):
        examples = [
            {
                "tokens": ["Bill", "was", "born", "in", "USA", "."],
                "entities": [
                    {"type": "per", "start": 0, "end": 1},
                    {"type": "loc", "start": 4, "end": 5},
                ],
                "relations": [{"type": "born in", "head": 0, "tail": 1}],
            },
            {
                "tokens": ["Jack", "Loves", "Mary", "."],
                "entities": [
                    {"type": "per", "start": 0, "end": 1},
                    {"type": "per", "start": 2, "end": 3},
                ],
                "relations": [{"type": "loves", "head": 0, "tail": 1}],
            },
        ]
        output = merge_examples(examples)
        expected = {
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
        self.assertEqual(expected, output)

    def test_merge_examples_single_example(self):
        examples = [
            {
                "tokens": ["Bill", "was", "born", "in", "USA", "."],
                "entities": [
                    {"type": "per", "start": 0, "end": 1},
                    {"type": "loc", "start": 4, "end": 5},
                ],
                "relations": [{"type": "born in", "head": 0, "tail": 1}],
            }
        ]
        output = merge_examples(examples)
        self.assertEqual(examples[0], output)

    def test_merge_examples_no_example(self):
        examples = []
        with self.assertRaises(ValueError) as ctx:
            merge_examples(examples)
        self.assertEqual(
            str(ctx.exception), "At least one example must be given to merge"
        )

    def test_texify_example_no_entities(self):
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
            "entities": [],
            "relations": [
                {"type": "born in", "head": 0, "tail": 1},
                {"type": "loves", "head": 2, "tail": 3},
            ],
        }
        output = texify_example(example, " ")
        expected = {
            "tokens": "Bill was born in USA . Jack Loves Mary .",
            "entities": [],
            "relations": [
                {"type": "born in", "head": 0, "tail": 1},
                {"type": "loves", "head": 2, "tail": 3},
            ],
        }
        self.assertEqual(output, expected)

    def test_texify_example_normal(self):
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
        output = texify_example(example, " ")
        expected = {
            "tokens": "Bill was born in USA . Jack Loves Mary .",
            "entities": [
                {"type": "per", "start": 0, "end": 4},
                {"type": "loc", "start": 17, "end": 20},
                {"type": "per", "start": 23, "end": 27},
                {"type": "per", "start": 34, "end": 38},
            ],
            "relations": [
                {"type": "born in", "head": 0, "tail": 1},
                {"type": "loves", "head": 2, "tail": 3},
            ],
        }
        self.assertEqual(output, expected)


if __name__ == "__main__":
    unittest.main()
