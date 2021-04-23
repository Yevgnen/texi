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
        train: bool = True,
    ):
        self.num_negative_entities = num_negative_entities
        self.num_negative_relations = num_negative_relations
        self.max_entity_length = max_entity_length
        self.negative_entity_type = negative_entity_type
        self.negative_relation_type = negative_relation_type
        self.is_train = train

    def train(self):
        self.is_train = True

    def eval(self):
        self.is_train = False

    def sample_negative_entities(self, example: Mapping) -> List[Dict]:
        text = example["tokens"]
        positives = example["entities"]
        positive_tuples = {(x["start"], x["end"]) for x in positives}
        negatives = []
        for length in range(1, self.max_entity_length + 1):
            for i in range(0, len(text) - length + 1):
                negative_tuple = (i, i + length)
                if self.is_train and negative_tuple in positive_tuples:
                    continue

                negative = {
                    "type": self.negative_entity_type,
                    "start": i,
                    "end": i + length,
                }
                negatives += [negative]

        if self.is_train:
            negatives = random.sample(
                negatives, min(len(negatives), self.num_negative_entities)
            )

        return negatives

    def sample_negative_relations(
        self, example: Mapping, entities: Optional[List[int]] = None
    ) -> List[Dict]:
        # No need to sample negative relations when evaluation.
        if not self.is_train:
            return []

        if entities is None:
            entities = example["entities"]

        if not entities:
            return []

        positives = example["relations"]
        positive_tuples = {(r["head"], r["tail"]) for r in positives}

        negatives = []
        for i, j in itertools.product(range(len(entities)), repeat=2):
            if i == j:
                continue

            if self.is_train and (i, j) in positive_tuples:
                continue

            negative = {"head": i, "tail": j, "type": self.negative_relation_type}
            negatives += [negative]

        if self.is_train:
            negatives = random.sample(
                negatives, min(len(negatives), self.num_negative_relations)
            )

        return negatives
