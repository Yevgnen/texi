# -*- coding: utf-8 -*-

from __future__ import annotations

import itertools
import random
from typing import TYPE_CHECKING, Dict, Iterable, List, Mapping, Optional, Union

import torch
from carton.collections import collate

from texi.preprocessing import LabelEncoder
from texi.pytorch.dataset import Dataset

if TYPE_CHECKING:
    from transformers import BertTokenizer, BertTokenizerFast


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
        entity_masks, entity_labels, entity_sample_masks = [], [], []
        max_length, max_entities = 0, 0
        for i, entities in enumerate(collated["entities"]):
            assert len(entities) > 0, "There must at least 1 negative entity."

            entities = collate(entities)

            num_tokens = len(collated["output"][i]["input_ids"])
            masks = create_span_mask(entities["start"], entities["end"], num_tokens)
            entity_masks += [masks]

            labels = torch.tensor(entities["label"], dtype=torch.int64)
            entity_labels += [labels]

            entity_sample_masks += [torch.ones(masks.size(0), dtype=torch.int64)]

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
        entity_sample_masks = [
            torch.nn.functional.pad(x, [0, max_entities - len(x)])
            for x in entity_sample_masks
        ]

        entity_masks = torch.stack(entity_masks)
        entity_labels = torch.stack(entity_labels)
        entity_sample_masks = torch.stack(entity_sample_masks)

        return {
            "entity_masks": entity_masks,
            "entity_labels": entity_labels,
            "entity_sample_masks": entity_sample_masks,
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
