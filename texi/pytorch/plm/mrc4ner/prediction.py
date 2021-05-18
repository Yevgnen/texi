# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import List, Tuple, Union, cast

import torch

from texi.apps.ner.utils import Entity
from texi.preprocessing import LabelEncoder

Scores = List[List[float]]
Spans = List[List[Tuple[int, int]]]
SpanWithScores = Tuple[Spans, Scores]
Entities = List[List[Entity]]
EntityWithScores = Tuple[Entities, Scores]


def predict_spans(
    span_label: torch.Tensor,
    span_index: torch.LongTensor,
    span_mask: torch.LongTensor,
    return_scores: bool = False,
) -> Union[Spans, tuple[Spans, Scores]]:
    if span_label.dtype == torch.float32:
        span_label = span_label.round()
        span_logit = span_label.tolist()
    elif span_label.dtype == torch.int64:
        if return_scores:
            raise ValueError("`return_score` must be False when span targets are given")
    else:
        raise TypeError(
            "`span_label` should `torch.int64` dtype when target is passed"
            " or `torch.float32` dtype when logit is passed"
        )

    span_label = span_label.masked_fill(span_mask, -1)
    span_labels = span_label.tolist()

    span_indices = span_index.tolist()

    spans, scores = [], []
    for i, sample_span_labels in enumerate(span_labels):
        sample_spans, sample_scores = [], []
        for j, (span, sample_span_label) in enumerate(
            zip(span_indices, sample_span_labels)
        ):
            if sample_span_label > 0:
                sample_spans += [span]
                if return_scores:
                    sample_scores += [span_logit[i][j]]
        spans += [sample_spans]
        scores += [sample_scores]

    if return_scores:
        return spans, scores

    return spans


def predict_entities(
    span_label: torch.Tensor,
    span_index: torch.LongTensor,
    span_mask: torch.LongTensor,
    offset_mapping: torch.LongTensor,
    entity_type: torch.LongTensor,
    entity_label_encoder: LabelEncoder,
    return_scores: bool = False,
) -> Union[Entities, EntityWithScores]:
    span_pred = predict_spans(
        span_label, span_index, span_mask, return_scores=return_scores
    )
    if return_scores:
        spans, span_scores = cast(SpanWithScores, span_pred)
    else:
        spans = cast(Spans, span_pred)

    offsets = offset_mapping.tolist()
    entity_types = entity_type.tolist()

    entities, scores = [], []
    for i, (sample_spans, sample_offsets, sample_entity_type) in enumerate(
        zip(spans, offsets, entity_types)
    ):
        sample_entities, sample_scores = [], []
        type_ = entity_label_encoder.decode_label(sample_entity_type)
        for j, (start, end) in enumerate(sample_spans):
            if sample_offsets[start][0] == 0 and sample_offsets[end][0] == 0:
                entity: Entity = {
                    "type": type_,
                    "start": 1,
                    "end": 1,
                }
                sample_entities += [entity]
                sample_scores += [span_scores[i][j]]
        entities += [sample_entities]
        scores += [sample_scores]

    if return_scores:
        return entities, scores

    return entities
