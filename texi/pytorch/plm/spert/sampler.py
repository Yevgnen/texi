# -*- coding: utf-8 -*-

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from texi.apps.ner.span_utils import sample_negative_entities, sample_negative_relations
from texi.utils import ModeKeys, PhaseMixin


class SpERTSampler(PhaseMixin):
    def __init__(
        self,
        num_negative_entities: int,
        num_negative_relations: int,
        max_entity_length: int,
        negative_entity_type: str = "NEGATIVE_ENTITY",
        negative_relation_type: str = "NEGATIVE_RELATION",
        mode: ModeKeys = ModeKeys.TRAIN,
    ) -> None:
        self.num_negative_entities = num_negative_entities
        self.num_negative_relations = num_negative_relations
        self.max_entity_length = max_entity_length
        self.negative_entity_type = negative_entity_type
        self.negative_relation_type = negative_relation_type
        self.mode = mode

    def sample_negative_entities(self, example: Mapping) -> list[dict[str, Any]]:
        return sample_negative_entities(
            example,
            self.max_entity_length,
            size=self.num_negative_entities,
            negative_entity_type=self.negative_entity_type,
        )

    def sample_negative_relations(self, example: Mapping) -> list[dict[str, Any]]:
        return sample_negative_relations(
            example,
            size=self.num_negative_relations,
            negative_relation_type=self.negative_relation_type,
        )
