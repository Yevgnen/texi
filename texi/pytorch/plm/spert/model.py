# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Union, cast

import torch
import torch.nn as nn
from transformers import BertModel

from texi.pytorch.masking import create_span_mask
from texi.pytorch.plm.pooling import get_pooling
from texi.pytorch.plm.utils import plm_path
from texi.pytorch.utils import split_apply


class SpERT(nn.Module):
    def __init__(
        self,
        bert: Union[BertModel, str],
        embedding_dim: int,
        num_entity_types: int,
        num_relation_types: int,
        negative_entity_index: int,
        max_entity_length: int = 100,
        max_relation_pairs: int = 1000,
        dropout: float = 0.2,
        global_context_pooling: str = "cls",
    ):
        super().__init__()
        if isinstance(bert, str):
            bert = BertModel.from_pretrained(plm_path(bert), add_pooling_layer=False)
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
        self.max_relation_pairs = max_relation_pairs
        self.dropout = nn.Dropout(p=dropout)
        self.global_context_pooling = get_pooling(global_context_pooling)

    def _mask_hidden_states(self, last_hidden_state, mask):
        # pylint: disable=no-self-use

        mask = mask.unsqueeze(dim=-1)
        masked = last_hidden_state.unsqueeze(dim=1) * mask

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
        relation,
        relation_context_mask,
        entity,
        entity_size,
        last_hidden_state,
    ):
        if relation.size(1) == 0:
            return relation.new_zeros(
                *relation.size()[:2], self.num_relation_types
            ).float()

        # relation_context: [B, R, L, H] -> [B, R, H]
        relation_context = self._mask_hidden_states(
            last_hidden_state, relation_context_mask
        )
        relation_context, _ = relation_context.max(dim=-2)

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
        context = self.global_context_pooling(bert_output, attention_mask)

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
    ) -> dict[str, torch.Tensor]:
        # last_hidden_state: [B, L, H]
        # entity_logit: [B, E, NE]
        # entity: [B, E, H]
        # entity_size: [B, E, D]
        last_hidden_state, entity_logit, entity, entity_size = self._forward_entity(
            input_ids, attention_mask, token_type_ids, entity_mask
        )

        # relation_logit: [B, R, NR]
        relation_logit = self._classify_relations(
            relations,
            relation_context_mask,
            entity,
            entity_size,
            last_hidden_state,
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
        # pylint: disable=no-self-use
        # NOTE: Filtered entities should have label -1.

        batch_size, max_entity_size = entity_label.size()
        device = entity_label.device

        index = torch.arange(batch_size, device=device)
        candidate = torch.arange(max_entity_size, device=device)
        candidate_index = torch.cartesian_prod(candidate, candidate).to(device)

        # Create relation pair candiates by Cartesian Product.
        # pair: [R, R]
        pair = entity_label[index[:, None], candidate_index.flatten()[None, :]].view(
            batch_size, -1, 2
        )

        # Filter invalid candidates:
        # 1. (i, i): by `diag_mask`.
        # 2. (i, j) with label_i < 0 *and* label_j < 0. We can not filter with
        # `or` condition because we still have to pad and stack
        # relations of each example in current batch.

        # pair: [B, R, 2]
        diag_mask = torch.eye(max_entity_size, device=device).flatten().bool()
        keep_mask = (pair >= 0).all(dim=-1).any(dim=0)
        pair_mask = ~diag_mask & keep_mask
        pair = pair[:, pair_mask, :]
        sample_mask = (pair >= 0).all(dim=-1).long()

        # TODO: Need a way to remove this corner case. Without this
        # test, the `min()` and `max()` call below will failed.
        if not pair_mask.any():
            pair = entity_label.new_zeros((batch_size, 0, 2))
            context = entity_label.new_zeros((batch_size, 0, max_length))
            sample_mask = entity_label.new_zeros((batch_size, 0))

            return pair, context, sample_mask

        # Create relation context mask.
        # Context start is defined as `min(head_end, tail_end)` and
        # context end is defined as `max(head_start, tail_start)`.
        # context_start, context_end: [B, R]
        candidate_index = candidate_index[pair_mask]
        entity_start, entity_end = entity_span[..., 0], entity_span[..., 1]
        context_start = (
            entity_end[index[:, None, None], candidate_index.unsqueeze(dim=0)]
            .min(dim=-1)[0]
            .flatten()
        )
        context_end = (
            entity_start[index[:, None, None], candidate_index.unsqueeze(dim=0)]
            .max(dim=-1)[0]
            .flatten()
        )

        context = create_span_mask(
            context_start, context_end, max_length, device=entity_label.device
        ).view(batch_size, -1, max_length)

        return pair, context, sample_mask

    def infer(
        self,
        input_ids: torch.LongTensor,  # [B, L]
        attention_mask: torch.LongTensor,  # [B, L]
        token_type_ids: torch.LongTensor,  # [B, L]
        entity_mask: torch.LongTensor,  # [B, E, L]
    ) -> dict[str, torch.Tensor]:
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
        if self.max_relation_pairs > 0:
            relation_logit = split_apply(
                self._classify_relations,
                [relation, relation_context_mask],
                self.max_relation_pairs,
                dim=1,
                entity=entity,
                entity_size=entity_size,
                last_hidden_state=last_hidden_state,
            )
        else:
            relation_logit = self._classify_relations(
                relation,
                relation_context_mask,
                entity,
                entity_size,
                last_hidden_state,
            )

        return {
            "entity_logit": entity_logit,
            "relation_logit": cast(torch.Tensor, relation_logit),
            "relation": relation,
            "relation_sample_mask": relation_sample_mask,
        }
