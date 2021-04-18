# -*- coding: utf-8 -*-

from __future__ import annotations

import itertools
import random
from typing import TYPE_CHECKING, Dict, Iterable, List, Mapping, Optional, Union

import torch
import torch.nn as nn
from carton.collections import collate
from torch.utils.data import Dataset

from texi.preprocessing import LabelEncoder
from texi.pytorch.dataset import Dataset

if TYPE_CHECKING:
    from transformers import BertModel, BertTokenizer, BertTokenizerFast


def create_span_mask(start: List[int], end: List[int], length: int) -> torch.LongTensor:
    if len(start) != len(end):
        raise ValueError(
            f"`start` and `end` should have same lengths: {len(start)} != {len(end)}"
        )

    if len(start) == 0:
        return torch.zeros((0, length), dtype=torch.int64)

    start = torch.tensor(start, dtype=torch.int64)
    end = torch.tensor(end, dtype=torch.int64)
    mask = torch.arange(length, dtype=torch.int64).unsqueeze(dim=-1)
    mask = (start <= mask) & (mask < end)
    mask = mask.transpose(0, 1).long()

    return mask


class SpERTSampler(object):
    def __init__(
        self,
        num_negative_entities: int,
        num_negative_relations: int,
        max_entity_length: int,
        negative_entity_type: str = "NOT_ENTITY",
        negative_relation_type: str = "NO_RELATION",
    ):
        self.num_negative_entities = num_negative_entities
        self.num_negative_relations = num_negative_relations
        self.max_entity_length = max_entity_length
        self.negative_entity_type = negative_entity_type
        self.negative_relation_type = negative_relation_type

    def sample_negative_entities(self, example: Mapping) -> List[Dict]:
        text = example["text"]
        positives = example["entities"]
        positive_tuples = {(x["start"], x["end"]) for x in positives}
        negatives = []
        for length in range(1, self.max_entity_length + 1):
            for i in range(0, len(text) - length + 1):
                mention = text[i : i + length]
                negative_tuple = (i, i + length)
                if negative_tuple not in positive_tuples:
                    negative = {
                        "mention": mention,
                        "type": self.negative_entity_type,
                        "start": i,
                        "end": i + length,
                    }
                    negatives += [negative]

        negatives = random.sample(
            negatives, min(len(negatives), self.num_negative_entities)
        )

        return negatives

    def sample_negative_relations(
        self,
        example: Mapping,
        entities: Optional[List[Mapping]] = None,
    ) -> List[Dict]:
        if entities is None:
            entities = example["entities"]

        if not entities:
            return []

        positives = example["relations"]
        positive_tuples = {
            (r["arg1"]["start"], r["arg1"]["end"], r["arg2"]["start"], r["arg2"]["end"])
            for r in positives
        }

        negatives = []
        for e1, e2 in itertools.product(entities, repeat=2):
            if e1 == e2:
                continue

            negative_tuple = (e1["start"], e1["end"], e2["start"], e2["end"])
            if negative_tuple in positive_tuples:
                continue

            negative = {"arg1": e1, "arg2": e2, "type": self.negative_relation_type}
            negatives += [negative]

        negatives = random.sample(
            negatives, min(len(negatives), self.num_negative_relations)
        )

        return negatives


class SpERTDataset(Dataset):
    def __init__(
        self,
        examples: Iterable[Dict],
        negative_sampler: SpERTSampler,
        entity_label_encoder: LabelEncoder,
        relation_label_encoder: LabelEncoder,
        tokenizer: Union[BertTokenizer, BertTokenizerFast] = None,
        train: bool = False,
    ):
        super().__init__(examples, tokenizer=tokenizer, train=train)
        self.negative_sampler = negative_sampler
        self.entity_label_encoder = entity_label_encoder
        self.relation_label_encoder = relation_label_encoder

    def _encode_entities(self, entities, tokens):
        offset, offsets = 0, []
        for token in tokens:
            offsets += [offset]
            offset += len(token)

        entity_indices = {}
        encoded_entities = []
        for i, entity in enumerate(entities):
            start = offsets[entity["start"] + 1]
            end = offsets[entity["end"] - 1 + 1] + len(tokens[entity["end"] - 1 + 1])
            encoded_entities += [
                {
                    "start": start,
                    "end": end,
                    "label": self.entity_label_encoder.encode_label(entity["type"]),
                }
            ]

            entity_indices[(entity["start"], entity["end"])] = i

        return encoded_entities, entity_indices

    def _encode_relations(self, relations, entity_indices):
        encoded_relations = []
        for rel in relations:
            head_index = entity_indices[rel["arg1"]["start"], rel["arg1"]["end"]]
            tail_index = entity_indices[rel["arg2"]["start"], rel["arg2"]["end"]]
            encoded_relations += [
                {
                    "head": head_index,
                    "tail": tail_index,
                    "label": self.relation_label_encoder.encode_label(rel["type"]),
                }
            ]

        return encoded_relations

    def encode_example(self, example, entities, relations):
        # Encode tokens.
        tokens = (
            [self.tokenizer.cls_token] + example["text"] + [self.tokenizer.sep_token]
        )
        output = self.tokenizer(tokens, add_special_tokens=False)

        # Encode entities.
        encoded_entities, entity_indices = self._encode_entities(
            entities, output["input_ids"]
        )

        # Encode relations.
        encoded_relations = self._encode_relations(relations, entity_indices)

        output = {k: list(itertools.chain.from_iterable(v)) for k, v in output.items()}

        return {
            "output": output,
            "entities": encoded_entities,
            "relations": encoded_relations,
        }

    def encode(self, example):
        # Collect entities.
        positive_entities = example["entities"]
        negative_entities = self.negative_sampler.sample_negative_entities(example)
        entities = positive_entities + negative_entities

        # Collect relations.
        positive_relations = example["relations"]
        negative_relations = self.negative_sampler.sample_negative_relations(
            example, positive_entities
        )
        relations = positive_relations + negative_relations

        return self.encode_example(example, entities, relations)

    def _collate_entities(self, collated):
        entity_masks, entity_labels = [], []
        max_length, max_entities = 0, 0
        for i, entities in enumerate(collated["entities"]):
            assert len(entities) > 0, "There must at least 1 negative entity."

            entities = collate(entities)

            num_tokens = len(collated["output"][i]["input_ids"])
            masks = create_span_mask(entities["start"], entities["end"], num_tokens)
            entity_masks += [masks]

            labels = torch.tensor(entities["label"], dtype=torch.int64)
            entity_labels += [labels]

            max_entities = max(max_entities, masks.size(0))
            max_length = max(max_length, masks.size(1))

        entity_masks = [
            torch.nn.functional.pad(
                x, [0, max_length - x.size(1), 0, max_entities - x.size(0)]
            )
            for x in entity_masks
        ]
        entity_labels = [
            torch.nn.functional.pad(x, [0, max_entities - len(x)])
            for x in entity_labels
        ]

        entity_masks = torch.stack(entity_masks)
        entity_labels = torch.stack(entity_labels)

        return {
            "entity_masks": entity_masks,
            "entity_labels": entity_labels,
        }

    def _collate_relations(self, collated):
        def _create_context_masks(heads, tails, entity_spans, length):
            head_starts, head_ends = zip(*[entity_spans[x] for x in heads])
            tail_starts, tail_ends = zip(*[entity_spans[x] for x in tails])
            starts, ends = [], []
            for hs, he, ts, te in zip(head_starts, head_ends, tail_starts, tail_ends):
                assert hs < he and ts < te and (he <= ts or te <= hs)
                if hs < ts:
                    starts += [he]
                    ends += [ts]
                else:
                    starts += [te]
                    ends += [hs]

            return create_span_mask(starts, ends, length)

        relation_args, relation_context_masks, relation_labels = [], [], []
        max_relations, max_length = 0, 0
        for output, entities, relations in zip(
            collated["output"], collated["entities"], collated["relations"]
        ):
            num_tokens = len(output["input_ids"])
            if len(relations) > 0:
                relations = collate(relations)
                heads = torch.tensor(relations["head"], dtype=torch.int64)
                tails = torch.tensor(relations["tail"], dtype=torch.int64)
                entity_spans = {
                    i: (x["start"], x["end"]) for i, x in enumerate(entities)
                }
                masks = _create_context_masks(
                    relations["head"], relations["tail"], entity_spans, num_tokens
                )
                args = torch.stack([heads, tails], dim=1)
                labels = torch.nn.functional.one_hot(
                    torch.tensor(relations["label"]), len(self.relation_label_encoder)
                )
            else:
                args = torch.zeros((0, 2), dtype=torch.int64)
                masks = torch.zeros((0, num_tokens), dtype=torch.int64)
                labels = torch.zeros(
                    (0, len(self.relation_label_encoder)), dtype=torch.int64
                )

            relation_args += [args]
            relation_context_masks += [masks]
            relation_labels += [labels]

            max_relations = max(max_relations, args.size(0))
            max_length = max(max_length, num_tokens)

        relation_args = [
            torch.nn.functional.pad(x, [0, 0, 0, max_relations - x.size(0)])
            for x in relation_args
        ]
        relation_context_masks = [
            torch.nn.functional.pad(
                x, [0, max_length - x.size(1), 0, max_relations - x.size(0)]
            )
            for x in relation_context_masks
        ]
        relation_labels = [
            torch.nn.functional.pad(x, [0, 0, 0, max_relations - x.size(0)])
            for x in relation_labels
        ]

        relation_args = torch.stack(relation_args)
        relation_context_masks = torch.stack(relation_context_masks)
        relation_labels = torch.stack(relation_labels)

        return {
            "relations": relation_args,
            "relation_context_masks": relation_context_masks,
            "relation_labels": relation_labels,
        }

    def collate(self, batch):
        batch = [self.encode(x) for x in batch]
        collated = collate(batch)
        entities = self._collate_entities(collated)
        relations = self._collate_relations(collated)
        output = collate(collated["output"])
        max_length = max(len(x) for x in output["input_ids"])
        output = {
            key: torch.stack(
                [
                    torch.nn.functional.pad(
                        torch.tensor(x, dtype=torch.int64), [0, max_length - len(x)]
                    )
                    for x in value
                ]
            )
            for key, value in output.items()
        }

        return {**output, **entities, **relations}


class SpERTLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.entity_loss = nn.CrossEntropyLoss(reduction="none")
        self.relation_loss = nn.BCEWithLogitsLoss(reduction="none")

    def _entity_loss(self, entity_logits, entity_labels, entity_sample_masks):
        entity_logits = entity_logits.transpose(-1, -2)
        entity_loss = self.entity_loss(entity_logits, entity_labels)
        entity_loss.masked_fill_(entity_sample_masks == 0, 0)
        entity_loss = entity_loss.sum() / entity_sample_masks.sum()

        return entity_loss

    def _relation_loss(self, relation_logits, relation_labels, relation_sample_masks):
        relation_loss = self.relation_loss(relation_logits, relation_labels.float())
        relation_loss.masked_fill_(relation_sample_masks.unsqueeze(-1) == 0, 0)
        relation_loss = relation_loss.sum() / relation_sample_masks.sum()

        return relation_loss

    def forward(
        self,
        entity_logits: torch.FloatTensor,
        entity_labels: torch.LongTensor,
        entity_sample_masks: torch.LongTensor,
        relation_logits: torch.FloatTensor,
        relation_labels: torch.LongTensor,
        relation_sample_masks: torch.LongTensor,
    ) -> torch.FloatTensor:
        entity_loss = self._entity_loss(
            entity_logits, entity_labels, entity_sample_masks
        )

        relation_loss = self._relation_loss(
            relation_logits, relation_labels, relation_sample_masks
        )

        loss = entity_loss + relation_loss

        return loss


class SpERT(nn.Module):
    def __init__(
        self,
        bert: BertModel,
        embedding_dim: int,
        num_entity_types: int,
        num_relation_types: int,
        max_entity_length: int = 100,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.bert = bert
        self.size_embedding = nn.Embedding(max_entity_length, embedding_dim)
        self.span_classifier = nn.Linear(
            embedding_dim + bert.config.hidden_size, num_entity_types
        )
        self.relation_classifier = nn.Linear(
            2 * embedding_dim + 3 * bert.config.hidden_size, num_relation_types
        )
        self.dropout = nn.Dropout(p=dropout)

    def _masked_hidden_states(self, hidden_states, masks):
        # pylint: disable=no-self-use

        masks = masks.unsqueeze(dim=-1)
        masked = hidden_states.unsqueeze(dim=1) * masks
        masked.masked_fill_(masks == 0, -1e20)

        return masked

    def _classify_entities(self, hidden_states, entity_masks, entity_sizes):
        # entities: [B, E, L, H] -> [B, E, H]
        entities = self._masked_hidden_states(hidden_states, entity_masks)
        entities, _ = entities.max(dim=-2)

        # entity_reprs: [B, E, H + D]
        entity_reprs = torch.cat([entities, entity_sizes], dim=-1)
        entity_reprs = self.dropout(entity_reprs)

        # entity_logits: [B, E, NE]
        entity_logits = self.span_classifier(entity_reprs)

        return entity_logits, entities

    def _classify_relations(
        self, hidden_states, relations, relation_context_masks, entities, entity_sizes
    ):
        # relation_contexts: [B, R, L, H] -> [B, R, H]
        relation_contexts = self._masked_hidden_states(
            hidden_states, relation_context_masks
        )
        relation_contexts, _ = relation_contexts.max(dim=-2)

        # entity_pairs: [B, R, 2H]
        batch_size, num_relations, _ = relations.size()
        entity_pairs = entities[
            torch.arange(batch_size).unsqueeze(dim=-1), relations.view(batch_size, -1)
        ].view(batch_size, num_relations, -1)

        # entity_pair_sizes: [B, R, 2E]
        entity_pair_sizes = entity_sizes[
            torch.arange(batch_size).unsqueeze(dim=-1), relations.view(batch_size, -1)
        ].view(batch_size, num_relations, -1)

        # relation_reprs: [B, R, 2E + 3H]
        relation_reprs = torch.cat(
            [relation_contexts, entity_pairs, entity_pair_sizes], dim=-1
        )
        relation_reprs = self.dropout(relation_reprs)

        # relation_logits: [B, R, NR]
        relation_logits = self.relation_classifier(relation_reprs)

        return relation_logits

    def forward(
        self,
        input_ids: torch.LongTensor,  # [B, L]
        attention_mask: torch.LongTensor,  # [B, L]
        token_type_ids: torch.LongTensor,  # [B, L]
        entity_masks: torch.LongTensor,  # [B, E, L]
        relations: torch.LongTensor,  # [B, R, 2]
        relation_context_masks: torch.LongTensor,  # [B, R, L]
    ) -> Dict[str, torch.FloatTensor]:
        # hidden_states: [B, L, H]
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden_states = bert_outputs.last_hidden_state

        # entity_sizes: [B, E, D]
        entity_sizes = self.size_embedding(entity_masks.sum(-1))

        # entity_reprs: [B, E, H + D]
        # entities: [B, E, H]
        entity_logits, entities = self._classify_entities(
            hidden_states, entity_masks, entity_sizes
        )

        # relation_logits: [B, R, NR]
        relation_logits = self._classify_relations(
            hidden_states, relations, relation_context_masks, entities, entity_sizes
        )

        return {
            "entity_logits": entity_logits,
            "relation_logits": relation_logits,
        }
