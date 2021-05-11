# -*- coding: utf-8 -*-

from __future__ import annotations

import itertools
import random
from collections.abc import Mapping
from typing import Any, Optional


def sample_negative_entities(
    example: Mapping,
    max_length: int,
    size: Optional[int] = None,
    negative_entity_type: str = "NEGATIVE_ENTITY",
) -> list[dict[str, Any]]:
    text = example["tokens"]
    positives = example["entities"]
    positive_tuples = {(x["start"], x["end"]) for x in positives}
    negatives = []
    for length in range(1, max_length + 1):
        for i in range(0, len(text) - length + 1):
            negative_tuple = (i, i + length)
            if negative_tuple in positive_tuples:
                continue

            negative = {
                "type": negative_entity_type,
                "start": i,
                "end": i + length,
            }
            negatives += [negative]

    if size is not None:
        negatives = random.sample(negatives, min(len(negatives), size))

    return negatives


def sample_negative_relations(
    example: Mapping,
    size: Optional[int] = None,
    negative_relation_type: str = "NEGATIVE_RELATION",
) -> list[dict[str, Any]]:
    entities = example["entities"]
    if not entities:
        return []

    positive_tuples = {(r["head"], r["tail"]) for r in example["relations"]}

    negatives = []
    for i, j in itertools.product(range(len(entities)), repeat=2):
        if i != j and (i, j) not in positive_tuples:
            negative = {
                "head": i,
                "tail": j,
                "type": negative_relation_type,
            }
            negatives += [negative]

    if size is not None:
        negatives = random.sample(negatives, min(len(negatives), size))

    return negatives
