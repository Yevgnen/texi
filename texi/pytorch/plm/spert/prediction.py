# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Tuple

import torch

from texi.preprocessing import LabelEncoder


def predict_entities(
    entity_logit: torch.Tensor,
    entity_sample_mask: torch.LongTensor,
    entity_span: torch.LongTensor,
    entity_label_encoder: LabelEncoder,
) -> List[List[Dict[str, Any]]]:
    if entity_logit.ndim == 3:
        entity_label = entity_logit.argmax(dim=-1)
    elif entity_logit.ndim == 2:
        entity_label = entity_logit
    else:
        raise ValueError("`entity_logit` should have 2 or 3 dimensions")

    entity_label.masked_fill_(~entity_sample_mask.bool(), -1).long()

    # Decode entities.
    entity_labels = entity_label.tolist()
    entity_spans = entity_span.tolist()

    entities = [
        [
            {
                "type": entity_label_encoder.decode_label(label),
                "start": entity_spans[i][j][0],
                "end": entity_spans[i][j][1],
            }
            for j, label in enumerate(labels)
            if label >= 0
        ]
        for i, labels in enumerate(entity_labels)
    ]

    return entities


def predict_relations(
    relation_logit: torch.Tensor,
    relation: torch.LongTensor,
    relation_sample_mask: torch.LongTensor,
    relation_label_encoder: LabelEncoder,
    relation_filter_threshold: float,
) -> List[Dict[str, Any]]:
    if relation_logit.dtype == torch.float32:
        relation_proba = torch.sigmoid(relation_logit)
    elif relation_logit.dtype == torch.int64:
        relation_proba = relation_logit
    else:
        raise TypeError(
            "`relation_proba` should `torch.int64` dtype when target is passed"
            " or `torch.float32` dtype when logit is passed"
        )

    if relation.size(1) < 1:
        return [[] for _ in range(len(relation))]

    filter_mask = relation_proba < relation_filter_threshold
    sample_mask = relation_sample_mask.unsqueeze(dim=-1).bool()
    relation_proba.masked_fill_(~sample_mask | filter_mask, -1)

    if relation.size(1) > 0:
        pairs = relation.tolist()
        relations = [
            [
                {
                    "type": relation_label_encoder.decode_label(k),
                    "head": pairs[i][j][0],
                    "tail": pairs[i][j][1],
                }
                for j, labels in enumerate(sample_labels)
                for k, label in enumerate(labels)
                if label >= 0
            ]
            for i, sample_labels in enumerate(relation_proba.tolist())
        ]

    return relations


def predict(
    entity_logit: torch.Tensor,
    entity_mask: torch.LongTensor,
    entity_span: torch.LongTensor,
    entity_label_encoder: LabelEncoder,
    negative_entity_index: int,
    relation_logit: torch.FloatTensor,
    relation_pair: torch.LongTensor,
    relation_sample_mask: torch.LongTensor,
    relation_label_encoder: LabelEncoder,
    negative_relation_index: int,
    relation_filter_threshold: float,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    # Predict entities.
    entities = predict_entities(
        entity_logit,
        entity_mask,
        entity_span,
        entity_label_encoder,
    )

    # Predict relation.
    relations = predict_relations(
        relation_logit,
        relation_pair,
        relation_sample_mask,
        relation_label_encoder,
        relation_filter_threshold,
    )

    negative_entity_label = entity_label_encoder.decode_label(negative_entity_index)
    negative_relation_label = relation_label_encoder.decode_label(
        negative_relation_index
    )

    def _normalize(entities, relations):
        new_entities = []
        entity_indices = {}
        for i, entity in enumerate(entities):
            if entity["type"] != negative_entity_label:
                entity_indices[i] = len(entity_indices)
                new_entities += [entity]

        new_relations = [
            {
                "type": r["type"],
                "head": entity_indices[r["head"]],
                "tail": entity_indices[r["tail"]],
            }
            for r in relations
            if r["type"] != negative_relation_label
        ]

        assert all(x["type"] != negative_entity_label for x in new_entities)
        assert all(
            x["type"] != negative_relation_label
            and new_entities[x["head"]]["type"] != negative_entity_label
            and new_entities[x["tail"]]["type"] != negative_entity_label
            for x in new_relations
        )

        return new_entities, new_relations

    return [
        _normalize(sample_entities, sample_relations)
        for sample_entities, sample_relations in zip(entities, relations)
    ]
