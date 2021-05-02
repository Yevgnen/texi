# -*- coding: utf-8 -*-

import itertools
from typing import Dict, Union, cast

import torch
import torch.nn as nn
from transformers import BertModel

from texi.pytorch.masking import create_span_mask
from texi.pytorch.plm.pooling import get_pooling
from texi.pytorch.plm.spert.dataset import stack_1d, stack_2d
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
            bert = BertModel.from_pretrained(bert)
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

        def _create_candidates(labels, spans):
            indices = (labels >= 0).nonzero(as_tuple=True)[0].tolist()
            spans = spans[indices].tolist()
            pairs = itertools.product(zip(indices, spans), repeat=2)

            outputs = []
            for (head, (head_start, head_end)), (tail, (tail_start, tail_end)) in pairs:
                # Ignore relations of overlapped entities.
                if head != tail and (head_start >= tail_end or tail_start >= head_end):
                    context = [min(head_end, tail_end), max(head_start, tail_start)]
                    pair = [head, tail]

                    outputs += [(context, pair, 1)]

            if not outputs:
                mask = entity_label.new_zeros((0, max_length))
                pair = entity_label.new_zeros((0, 2))
                sample_mask = entity_label.new_zeros((0,))
            else:
                outputs = [entity_label.new_tensor(x) for x in zip(*outputs)]
                context, pair, sample_mask = outputs
                mask = create_span_mask(
                    context[:, 0], context[:, 1], max_length, device=entity_label.device
                )

            return mask, pair, sample_mask

        masks, pairs, sample_masks = zip(
            *[
                _create_candidates(sample_label, sample_span)
                for sample_label, sample_span in zip(entity_label, entity_span)
            ]
        )

        max_relations = max(len(x) for x in pairs)
        mask = stack_2d(masks, max_relations, max_length)
        relation = stack_2d(pairs, max_relations, 2)
        sample_mask = stack_1d(sample_masks, max_relations)

        return relation, mask, sample_mask

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
                last_hidden_state,
                relation,
                relation_context_mask,
                entity,
                entity_size,
            )

        return {
            "entity_logit": entity_logit,
            "relation_logit": cast(torch.Tensor, relation_logit),
            "relation": relation,
            "relation_sample_mask": relation_sample_mask,
        }
