# -*- coding: utf-8 -*-

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from texi.apps.ner.span_utils import sample_negative_entities, sample_negative_relations


class SpERTSampler(object):
    def __init__(
        self,
        num_negative_entities: int,
        num_negative_relations: int,
        max_entity_length: int,
        negative_entity_type: str = "NOT_ENTITY",
        negative_relation_type: str = "NO_RELATION",
        train: bool = True,
    ) -> None:
        self.num_negative_entities = num_negative_entities
        self.num_negative_relations = num_negative_relations
        self.max_entity_length = max_entity_length
        self.negative_entity_type = negative_entity_type
        self.negative_relation_type = negative_relation_type
        self.is_train = train

    def train(self) -> None:
        self.is_train = True

    def eval(self) -> None:
        self.is_train = False

    def sample_negative_entities(self, example: Mapping) -> list[dict[str, Any]]:
        size = self.num_negative_entities if self.is_train else None

        return sample_negative_entities(
            example,
            self.max_entity_length,
            size=size,
            negative_entity_type=self.negative_entity_type,
        )

    def sample_negative_relations(self, example: Mapping) -> list[dict[str, Any]]:
        # No need to sample negative relations when evaluation.
        if not self.is_train:
            return []

        size = self.num_negative_relations if self.is_train else None

        return sample_negative_relations(
            example,
            size=self.num_negative_relations,
            negative_relation_type=self.negative_relation_type,
        )
