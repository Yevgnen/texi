# -*- coding: utf-8 -*-

from __future__ import annotations

import itertools
import random
from typing import Callable, Optional

from texi.apps.ner.utils import Entity, NerExample, Relation

from texi.preprocessing import LabelEncoder


def sample_negative_entities(
    example: NerExample,
    max_length: int,
    size: Optional[int] = None,
    negative_entity_type: str = "NEGATIVE_ENTITY",
    predicate: Optional[Callable[[int, int], bool]] = None,
) -> list[Entity]:
    text = example["tokens"]
    positives = example["entities"]

    if callable(predicate):
        _predicate = predicate
    else:
        positive_tuples = {(x["start"], x["end"]) for x in positives}

        def _predicate(start, end):
            return (start, end) not in positive_tuples

    negatives = []
    for length in range(1, max_length + 1):
        for i in range(0, len(text) - length + 1):
            if _predicate(i, i + length):
                negative: Entit = {
                    "type": negative_entity_type,
                    "start": i,
                    "end": i + length,
                }
                negatives += [negative]

    if size is not None:
        negatives = random.sample(negatives, min(len(negatives), size))

    return negatives


def sample_negative_relations(
    example: NerExample,
    size: Optional[int] = None,
    negative_relation_type: str = "NEGATIVE_RELATION",
    predicate: Optional[Callable[[int, int], bool]] = None,
) -> list[Relation]:
    entities = example["entities"]
    if not entities:
        return []

    if callable(predicate):
        _predicate = predicate
    else:
        positive_tuples = {(r["head"], r["tail"]) for r in example["relations"]}

        def _predicate(head, tail):
            return head != tail and (head, tail) not in positive_tuples

    negatives = []
    for i, j in itertools.product(range(len(entities)), repeat=2):
        if _predicate(i, j):
            negative: Relation = {
                "head": i,
                "tail": j,
                "type": negative_relation_type,
            }
            negatives += [negative]

    if size is not None:
        negatives = random.sample(negatives, min(len(negatives), size))

    return negatives


class SpanNerNegativeSampler(object):
    def __init__(
        self,
        num_negative_entities: int,
        num_negative_relations: int,
        max_entity_length: int,
        negative_entity_type: str = "NEGATIVE_ENTITY",
        negative_relation_type: str = "NEGATIVE_RELATION",
        relation_argument_types: Optional[Mapping] = None,
    ) -> None:
        self.num_negative_entities = num_negative_entities
        self.num_negative_relations = num_negative_relations
        self.max_entity_length = max_entity_length
        self.negative_entity_type = negative_entity_type
        self.negative_relation_type = negative_relation_type
        self.relation_argument_types = relation_argument_types

    def sample_negative_entities(self, example: NerExample) -> list[Entity]:
        return sample_negative_entities(
            example,
            self.max_entity_length,
            size=self.num_negative_entities,
            negative_entity_type=self.negative_entity_type,
        )

    def sample_negative_relations(self, example: NerExample) -> list[Relation]:
        if self.relation_argument_types:

            def predicate(head, tail):
                return any(
                    r["head"] == example["entities"][head]["type"]
                    and r["tail"] == example["entities"][tail]["type"]
                    for r in self.relation_argument_types.values()
                )

        else:
            predicate = None  # type: ignore

        return sample_negative_relations(
            example,
            size=self.num_negative_relations,
            negative_relation_type=self.negative_relation_type,
            predicate=predicate,
        )


class RelationFilter(object):
    def __init__(
        self,
        relation_argument_types: Mapping,
        entity_label_encoder: LabelEncoder,
        relation_label_encoder: LabelEncoder,
    ) -> None:
        self.types = relation_argument_types
        self.type_ids = {
            relation_label_encoder.encode_label(r): {
                "head": entity_label_encoder.encode_label(args["head"]),
                "tail": entity_label_encoder.encode_label(args["tail"]),
            }
            for r, args in relation_argument_types.items()
        }
