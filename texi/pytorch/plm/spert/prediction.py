# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Tuple

import torch

from texi.preprocessing import LabelEncoder


def decode_relations(
    relation_prob: torch.Tensor,
    relation: torch.LongTensor,
    relation_sample_mask: torch.LongTensor,
    relation_label_encoder: LabelEncoder,
    negative_relation_index: int,
    relation_filter_threshold: float,
) -> List[List[Dict[str, Any]]]:
    relation_filter_mask = relation_prob < relation_filter_threshold
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
    relation_logit: torch.FloatTensor,
    relation: torch.LongTensor,
    relation_sample_mask: torch.LongTensor,
    relation_label_encoder: LabelEncoder,
    negative_relation_index: int,
    relation_filter_threshold: float,
) -> List[Dict[str, Any]]:
    if relation_logit.size(1) > 0:
        relation_predictions = decode_relations(
            torch.sigmoid(relation_logit),
            relation,
            relation_sample_mask,
            relation_label_encoder,
            negative_relation_index,
            relation_filter_threshold,
        )
    else:
        relation_predictions = [[] for _ in range(len(relation))]

    return relation_predictions


def predict(
    entity_logit: torch.FloatTensor,
    entity_mask: torch.LongTensor,
    entity_span: torch.LongTensor,
    entity_label_encoder: LabelEncoder,
    negative_entity_index: int,
    relation_logit: torch.FloatTensor,
    relation: torch.LongTensor,
    relation_sample_mask: torch.LongTensor,
    relation_label_encoder: LabelEncoder,
    negative_relation_index: int,
    relation_filter_threshold: float,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    # Predict entities.
    entity_prediction = predict_entities(
        entity_logit,
        entity_mask,
        entity_span,
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
        relation_filter_threshold,
    )

    return list(zip(entity_prediction, relation_predictions))
