# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Tuple

import torch

from texi.preprocessing import LabelEncoder


def decode_entities(
    entity_labels: torch.LongTensor,
    entity_sample_masks: torch.LongTensor,
    entity_token_spans: torch.LongTensor,
    entity_label_encoder: LabelEncoder,
    negative_entity_index: int,
    filter_negatives: bool = True,
) -> List[List[Dict[str, Any]]]:
    non_entity_masks = entity_labels == negative_entity_index
    entity_labels = entity_labels.masked_fill(
        ~entity_sample_masks | non_entity_masks, -1
    ).long()
    entity_token_spans = entity_token_spans.cpu().numpy().tolist()

    entities = [
        [
            {
                "type": entity_label_encoder.decode_label(
                    label if label >= 0 else negative_entity_index
                ),
                "start": entity_token_spans[i][j][0],
                "end": entity_token_spans[i][j][1],
            }
            for j, label in enumerate(labels)
            if not filter_negatives or label >= 0
        ]
        for i, labels in enumerate(entity_labels.detach().cpu().numpy().tolist())
    ]

    return entities


def decode_relations(
    relation_labels: torch.LongTensor,
    relations: torch.LongTensor,
    relation_sample_masks: torch.LongTensor,
    relation_label_encoder: LabelEncoder,
    negative_relation_index: int,
    filter_negatives: bool = True,
) -> List[List[Dict[str, Any]]]:
    no_relation_masks = relation_labels == negative_relation_index
    relation_labels = relation_labels.masked_fill(
        ~relation_sample_masks.bool() | no_relation_masks, -1
    ).long()

    if relations.size(1) > 0:
        relations = relations.detach().cpu().numpy().tolist()
        relations = [
            [
                {
                    "type": relation_label_encoder.decode_label(
                        label if label >= 0 else negative_relation_index
                    ),
                    "head": relations[i][j][0],
                    "tail": relations[i][j][1],
                }
                for j, label in enumerate(labels)
                if not filter_negatives or label >= 0
            ]
            for i, labels in enumerate(relation_labels.detach().cpu().numpy().tolist())
        ]
    else:
        relations = [[] for _ in range(len(relations))]

    return relations


def predict_entities(
    entity_logits: torch.FloatTensor,
    entity_masks: torch.LongTensor,
    entity_token_spans: torch.LongTensor,
    entity_label_encoder: LabelEncoder,
    negative_entity_index: int,
    filter_negatives: bool = True,
) -> List[Dict[str, Any]]:
    # Predict entity labels.
    entity_sample_masks = entity_masks.sum(dim=-1) > 0
    entity_labels = entity_logits.argmax(dim=-1)

    # Decode entities.
    entity_predictions = decode_entities(
        entity_labels,
        entity_sample_masks,
        entity_token_spans,
        entity_label_encoder,
        negative_entity_index,
        filter_negatives=filter_negatives,
    )

    return entity_predictions


def predict_relations(
    relation_logits: torch.FloatTensor,
    relations: torch.LongTensor,
    relation_sample_masks: torch.LongTensor,
    relation_label_encoder: LabelEncoder,
    negative_relation_index: int,
    filter_negatives: bool = True,
) -> List[Dict[str, Any]]:
    if relation_logits.size(1) > 0:
        # Predict relation labels.
        relation_labels = relation_logits.argmax(dim=-1)

        # Decode relations.
        relation_predictions = decode_relations(
            relation_labels,
            relations,
            relation_sample_masks,
            relation_label_encoder,
            negative_relation_index,
            filter_negatives=filter_negatives,
        )
    else:
        relation_predictions = [[] for _ in range(len(relations))]

    return relation_predictions


def predict(
    entity_logits: torch.FloatTensor,
    entity_masks: torch.LongTensor,
    entity_token_spans: torch.LongTensor,
    entity_label_encoder: LabelEncoder,
    negative_entity_index: int,
    relation_logits: torch.FloatTensor,
    relations: torch.LongTensor,
    relation_sample_masks: torch.LongTensor,
    relation_label_encoder: LabelEncoder,
    negative_relation_index: int,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    # Predict entities.
    entity_predictions = predict_entities(
        entity_logits,
        entity_masks,
        entity_token_spans,
        entity_label_encoder,
        negative_entity_index,
        filter_negatives=True,
    )
    for example_entity_predictions in entity_predictions:
        example_entity_predictions.sort(key=lambda x: x["start"])

    # Predict relations.
    relation_predictions = predict_relations(
        relation_logits,
        relations,
        relation_sample_masks,
        relation_label_encoder,
        negative_relation_index,
        filter_negatives=True,
    )

    return list(zip(entity_predictions, relation_predictions))
