# -*- coding: utf-8 -*-

import itertools
import random
from typing import Dict, List, Mapping, Optional


class SpERTSampler(object):
    def __init__(
        self,
        num_negative_entities: int,
        num_negative_relations: int,
        max_entity_length: int,
        negative_entity_type: str = "NOT_ENTITY",
        negative_relation_type: str = "NO_RELATION",
    ):
        self.num_negative_entities = num_negative_entities
        self.num_negative_relations = num_negative_relations
        self.max_entity_length = max_entity_length
        self.negative_entity_type = negative_entity_type
        self.negative_relation_type = negative_relation_type

    def sample_negative_entities(self, example: Mapping) -> List[Dict]:
        text = example["text"]
        positives = example["entities"]
        positive_tuples = {(x["start"], x["end"]) for x in positives}
        negatives = []
        for length in range(1, self.max_entity_length + 1):
            for i in range(0, len(text) - length + 1):
                mention = text[i : i + length]
                negative_tuple = (i, i + length)
                if negative_tuple not in positive_tuples:
                    negative = {
                        "mention": mention,
                        "type": self.negative_entity_type,
                        "start": i,
                        "end": i + length,
                    }
                    negatives += [negative]

        negatives = random.sample(
            negatives, min(len(negatives), self.num_negative_entities)
        )

        return negatives

    def sample_negative_relations(
        self,
        example: Mapping,
        entities: Optional[List[Mapping]] = None,
    ) -> List[Dict]:
        if entities is None:
            entities = example["entities"]

        if not entities:
            return []

        positives = example["relations"]
        positive_tuples = {
            (r["arg1"]["start"], r["arg1"]["end"], r["arg2"]["start"], r["arg2"]["end"])
            for r in positives
        }

        negatives = []
        for e1, e2 in itertools.product(entities, repeat=2):
            if e1 == e2:
                continue

            negative_tuple = (e1["start"], e1["end"], e2["start"], e2["end"])
            if negative_tuple in positive_tuples:
                continue

            negative = {"arg1": e1, "arg2": e2, "type": self.negative_relation_type}
            negatives += [negative]

        negatives = random.sample(
            negatives, min(len(negatives), self.num_negative_relations)
        )

        return negatives
