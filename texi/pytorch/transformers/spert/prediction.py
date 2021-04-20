# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Tuple

import torch

from texi.preprocessing import LabelEncoder


def decode_entities(
    entity_label: torch.LongTensor,
    entity_sample_mask: torch.LongTensor,
    entity_token_span: torch.LongTensor,
    entity_label_encoder: LabelEncoder,
    negative_entity_index: int,
    filter_negatives: bool = True,
) -> List[List[Dict[str, Any]]]:
    negative_entity_mask = entity_label == negative_entity_index
    entity_label = entity_label.masked_fill(
        ~entity_sample_mask | negative_entity_mask, -1
    ).long()
    entity_token_span = entity_token_span.cpu().numpy().tolist()

    entities = [
        [
            {
                "type": entity_label_encoder.decode_label(
                    label if label >= 0 else negative_entity_index
                ),
                "start": entity_token_span[i][j][0],
                "end": entity_token_span[i][j][1],
            }
            for j, label in enumerate(labels)
            if not filter_negatives or label >= 0
        ]
        for i, labels in enumerate(entity_label.detach().cpu().numpy().tolist())
    ]

    return entities


def decode_relations(
    relation_label: torch.LongTensor,
    relation: torch.LongTensor,
    relation_sample_mask: torch.LongTensor,
    relation_label_encoder: LabelEncoder,
    negative_relation_index: int,
    filter_negatives: bool = True,
) -> List[List[Dict[str, Any]]]:
    negative_relation_mask = relation_label == negative_relation_index
    relation_label = relation_label.masked_fill(
        ~relation_sample_mask.bool() | negative_relation_mask, -1
    ).long()

    if relation.size(1) > 0:
        relation = relation.detach().cpu().numpy().tolist()
        relation = [
            [
                {
                    "type": relation_label_encoder.decode_label(
                        label if label >= 0 else negative_relation_index
                    ),
                    "head": relation[i][j][0],
                    "tail": relation[i][j][1],
                }
                for j, label in enumerate(labels)
                if not filter_negatives or label >= 0
            ]
            for i, labels in enumerate(relation_label.detach().cpu().numpy().tolist())
        ]
    else:
        relation = [[] for _ in range(len(relation))]

    return relation


def predict_entities(
    entity_logit: torch.FloatTensor,
    entity_mask: torch.LongTensor,
    entity_token_span: torch.LongTensor,
    entity_label_encoder: LabelEncoder,
    negative_entity_index: int,
    filter_negatives: bool = True,
) -> List[List[Dict[str, Any]]]:
    # Predict entity labels.
    entity_sample_mask = entity_mask.sum(dim=-1) > 0
    entity_label = entity_logit.argmax(dim=-1)

    # Decode entities.
    entity_prediction = decode_entities(
        entity_label,
        entity_sample_mask,
        entity_token_span,
        entity_label_encoder,
        negative_entity_index,
        filter_negatives=filter_negatives,
    )

    return entity_prediction


def predict_relations(
    relation_logit: torch.FloatTensor,
    relation: torch.LongTensor,
    relation_sample_mask: torch.LongTensor,
    relation_label_encoder: LabelEncoder,
    negative_relation_index: int,
    filter_negatives: bool = True,
) -> List[Dict[str, Any]]:
    if relation_logit.size(1) > 0:
        # Predict relation labels.
        relation_label = relation_logit.argmax(dim=-1)

        # Decode relation.
        relation_predictions = decode_relations(
            relation_label,
            relation,
            relation_sample_mask,
            relation_label_encoder,
            negative_relation_index,
            filter_negatives=filter_negatives,
        )
    else:
        relation_predictions = [[] for _ in range(len(relation))]

    return relation_predictions


def predict(
    entity_logit: torch.FloatTensor,
    entity_mask: torch.LongTensor,
    entity_token_span: torch.LongTensor,
    entity_label_encoder: LabelEncoder,
    negative_entity_index: int,
    relation_logit: torch.FloatTensor,
    relation: torch.LongTensor,
    relation_sample_mask: torch.LongTensor,
    relation_label_encoder: LabelEncoder,
    negative_relation_index: int,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    # Predict entities.
    entity_prediction = predict_entities(
        entity_logit,
        entity_mask,
        entity_token_span,
        entity_label_encoder,
        negative_entity_index,
        filter_negatives=True,
    )
    for example_entity_prediction in entity_prediction:
        example_entity_prediction.sort(key=lambda x: x["start"])

    # Predict relation.
    relation_predictions = predict_relations(
        relation_logit,
        relation,
        relation_sample_mask,
        relation_label_encoder,
        negative_relation_index,
        filter_negatives=True,
    )

    return list(zip(entity_prediction, relation_predictions))
