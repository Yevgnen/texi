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
) -> List[List[Dict[str, Any]]]:
    entity_label = entity_label.masked_fill(~entity_sample_mask, -1).long()

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
            if label >= 0
        ]
        for i, labels in enumerate(entity_label.detach().cpu().numpy().tolist())
    ]

    return entities


def decode_relations(
    relation_prob: torch.Tensor,
    relation: torch.LongTensor,
    relation_sample_mask: torch.LongTensor,
    relation_label_encoder: LabelEncoder,
    negative_relation_index: int,
) -> List[List[Dict[str, Any]]]:
    relation_filter_mask = relation_prob < 0.4
    relation_pad_mask = ~relation_sample_mask.unsqueeze(dim=-1).bool()
    mask = relation_pad_mask | relation_filter_mask
    relation_prob = relation_prob.masked_fill(mask, -1)

    if relation.size(1) > 0:
        relation = relation.detach().cpu().numpy().tolist()
        relation = [
            [
                {
                    "type": relation_label_encoder.decode_label(
                        k if label >= 0 else negative_relation_index
                    ),
                    "head": relation[i][j][0],
                    "tail": relation[i][j][1],
                }
                for j, labels in enumerate(sample_labels)
                for k, label in enumerate(labels)
                if label >= 0
            ]
            for i, sample_labels in enumerate(
                relation_prob.detach().cpu().numpy().tolist()
            )
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
    )

    return entity_prediction


def predict_relations(
    relation_logit: torch.FloatTensor,
    relation: torch.LongTensor,
    relation_sample_mask: torch.LongTensor,
    relation_label_encoder: LabelEncoder,
    negative_relation_index: int,
) -> List[Dict[str, Any]]:
    if relation_logit.size(1) > 0:
        relation_predictions = decode_relations(
            torch.sigmoid(relation_logit),
            relation,
            relation_sample_mask,
            relation_label_encoder,
            negative_relation_index,
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
    )

    return list(zip(entity_prediction, relation_predictions))
