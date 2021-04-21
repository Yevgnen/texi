# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import torch
import torch.nn as nn

from texi.pytorch.masking import create_span_mask
from texi.pytorch.transformers.pooling import cls_pooling

if TYPE_CHECKING:
    from transformers import BertModel


class SpERT(nn.Module):
    def __init__(
        self,
        bert: BertModel,
        embedding_dim: int,
        num_entity_types: int,
        num_relation_types: int,
        negative_entity_index: int,
        max_entity_length: int = 100,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.bert = bert
        self.size_embedding = nn.Embedding(max_entity_length, embedding_dim)
        self.span_classifier = nn.Linear(
            embedding_dim + 2 * bert.config.hidden_size, num_entity_types
        )
        self.relation_classifier = nn.Linear(
            2 * embedding_dim + 3 * bert.config.hidden_size, num_relation_types
        )
        self.num_entity_types = num_entity_types
        self.num_relation_types = num_relation_types
        self.negative_entity_index = negative_entity_index
        self.max_entity_length = max_entity_length
        self.dropout = nn.Dropout(p=dropout)

    def _mask_hidden_states(self, last_hidden_state, mask):
        # pylint: disable=no-self-use

        mask = mask.unsqueeze(dim=-1)
        masked = last_hidden_state.unsqueeze(dim=1) * mask
        masked.masked_fill_(mask == 0, -1e20)

        return masked

    def _classify_entities(self, last_hidden_state, context, entity_mask, entity_size):
        # entity: [B, E, L, H] -> [B, E, H]
        entity = self._mask_hidden_states(last_hidden_state, entity_mask)
        entity, _ = entity.max(dim=-2)

        # entity_repr: [B, E, 2H + D]
        context = context.unsqueeze(dim=1).repeat(1, entity.size(1), 1)
        entity_repr = torch.cat([entity, context, entity_size], dim=-1)
        entity_repr = self.dropout(entity_repr)

        # entity_logit: [B, E, NE]
        entity_logit = self.span_classifier(entity_repr)

        return entity_logit, entity

    def _classify_relations(
        self,
        last_hidden_state,
        relation,
        relation_context_mask,
        relation_sample_mask,
        entity,
        entity_size,
    ):
        if relation.size(1) == 0:
            return relation.new_zeros(*relation.size()[:2], self.num_relation_types)

        # relation_context: [B, R, L, H] -> [B, R, H]
        relation_context = self._mask_hidden_states(
            last_hidden_state, relation_context_mask
        )
        relation_context, _ = relation_context.max(dim=-2)
        relation_context.masked_fill_(relation_sample_mask.unsqueeze(dim=-1) == 1, 0)

        # entity_pair: [B, R, 2H]
        batch_size, num_relations, _ = relation.size()
        entity_pair = entity[
            torch.arange(batch_size).unsqueeze(dim=-1), relation.view(batch_size, -1)
        ].view(batch_size, num_relations, -1)

        # entity_pair_size: [B, R, 2E]
        entity_pair_size = entity_size[
            torch.arange(batch_size).unsqueeze(dim=-1), relation.view(batch_size, -1)
        ].view(batch_size, num_relations, -1)

        # relation_repr: [B, R, 2E + 3H]
        relation_repr = torch.cat(
            [relation_context, entity_pair, entity_pair_size], dim=-1
        )
        relation_repr = self.dropout(relation_repr)

        # relation_logit: [B, R, NR]
        relation_logit = self.relation_classifier(relation_repr)

        return relation_logit

    def _forward_entity(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        entity_mask,
    ):
        # last_hidden_state: [B, L, H]
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        last_hidden_state = bert_output.last_hidden_state

        # context: [B, H]
        context = cls_pooling(bert_output, attention_mask)

        # entity_size: [B, E, D]
        entity_size = self.size_embedding(entity_mask.sum(-1))

        # entity_logit: [B, E, NE]
        # entity: [B, E, H]
        entity_logit, entity = self._classify_entities(
            last_hidden_state, context, entity_mask, entity_size
        )

        return last_hidden_state, entity_logit, entity, entity_size

    def forward(
        self,
        input_ids: torch.LongTensor,  # [B, L]
        attention_mask: torch.LongTensor,  # [B, L]
        token_type_ids: torch.LongTensor,  # [B, L]
        entity_mask: torch.LongTensor,  # [B, E, L]
        relations: torch.LongTensor,  # [B, R, 2]
        relation_context_mask: torch.LongTensor,  # [B, R, L]
        relation_sample_mask: torch.LongTensor,  # [B, R]
    ) -> Dict[str, torch.FloatTensor]:
        # last_hidden_state: [B, L, H]
        # entity_logit: [B, E, NE]
        # entity: [B, E, H]
        # entity_size: [B, E, D]
        last_hidden_state, entity_logit, entity, entity_size = self._forward_entity(
            input_ids, attention_mask, token_type_ids, entity_mask
        )

        # relation_logit: [B, R, NR]
        relation_logit = self._classify_relations(
            last_hidden_state,
            relations,
            relation_context_mask,
            relation_sample_mask,
            entity,
            entity_size,
        )

        return {
            "entity_logit": entity_logit,
            "relation_logit": relation_logit,
        }

    def _filter_entities(self, entity_logit, entity_mask):
        # Non-entiies and padding will have label -1.

        # entity_label: [B, E]
        entity_sample_mask = entity_mask.sum(dim=-1) > 0
        entity_label = entity_logit.argmax(dim=-1)
        negative_entity_mask = entity_label == self.negative_entity_index
        entity_label.masked_fill_(~entity_sample_mask | negative_entity_mask, -1)

        # entity_span: [B, E, 2]
        start = entity_mask.argmax(dim=-1, keepdim=True)
        end = entity_mask.sum(dim=-1, keepdim=True) + start
        entity_span = torch.cat((start, end), dim=-1)

        return entity_label, entity_span

    def _filter_relations(self, entity_label, entity_span, max_length):
        relations, relation_context_masks, relation_sample_masks = [], [], []
        max_relations = 0
        for i, labels in enumerate(entity_label):
            pairs, context_starts, context_ends = [], [], []
            indices = entity_label.new_tensor(range(len(labels)))
            for head, tail in torch.cartesian_prod(indices, indices):
                if labels[head] >= 0 and labels[tail] >= 0 and head != tail:
                    head_start, head_end = entity_span[i][head]
                    tail_start, tail_end = entity_span[i][tail]

                    # Ignore relations of overlapped entities.
                    if head_start >= tail_end or tail_start >= head_end:
                        start = min(head_end, tail_end)
                        end = max(head_start, tail_start)
                        context_starts += [start]
                        context_ends += [end]
                        pairs += [(head, tail)]

            assert len(pairs) == len(context_starts) == len(context_ends)

            relation_context_masks += [
                create_span_mask(
                    context_starts,
                    context_ends,
                    max_length,
                    device=entity_label.device,
                )
            ]
            relation_sample_masks += [entity_label.new_ones(len(pairs))]
            if len(pairs) > 0:
                relations += [entity_label.new_tensor(pairs)]
            else:
                relations += [entity_label.new_zeros((0, 2))]
            max_relations = max(max_relations, len(pairs))

        relations = [
            torch.nn.functional.pad(x, [0, 0, 0, max_relations - len(x)])
            for x in relations
        ]
        relation_context_masks = [
            torch.nn.functional.pad(x, [0, 0, 0, max_relations - len(x)])
            for x in relation_context_masks
        ]
        relation_sample_masks = [
            torch.nn.functional.pad(x, [0, max_relations - x.size(0)])
            for x in relation_sample_masks
        ]

        relation = torch.stack(relations)
        relation_context_mask = torch.stack(relation_context_masks)
        relation_sample_mask = torch.stack(relation_sample_masks)

        return relation, relation_context_mask, relation_sample_mask

    def infer(
        self,
        input_ids: torch.LongTensor,  # [B, L]
        attention_mask: torch.LongTensor,  # [B, L]
        token_type_ids: torch.LongTensor,  # [B, L]
        entity_mask: torch.LongTensor,  # [B, E, L]
    ) -> Dict[str, torch.Tensor]:
        # last_hidden_state: [B, L, H]
        # entity_logit: [B, E, NE]
        # entity: [B, E, H]
        # entity_size: [B, E, D]
        last_hidden_state, entity_logit, entity, entity_size = self._forward_entity(
            input_ids, attention_mask, token_type_ids, entity_mask
        )

        # entity_label: [B, E]
        # entity_span: [B, E, 2]
        entity_label, entity_span = self._filter_entities(entity_logit, entity_mask)

        # relation: [B, R, 2]
        # relation_context_mask: [B, R, L]
        # relation_sample_mask: [B, R]
        (
            relation,
            relation_context_mask,
            relation_sample_mask,
        ) = self._filter_relations(entity_label, entity_span, input_ids.size(1))

        # relation_logit: [B, R, NR]
        relation_logit = self._classify_relations(
            last_hidden_state,
            relation,
            relation_context_mask,
            relation_sample_mask,
            entity,
            entity_size,
        )

        return {
            "entity_logit": entity_logit,
            "relation_logit": relation_logit,
            "relation": relation,
            "relation_sample_mask": relation_sample_mask,
        }
