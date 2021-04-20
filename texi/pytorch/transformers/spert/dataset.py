# -*- coding: utf-8 -*-

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Dict, Iterable, List, Union

import torch
from carton.collections import collate

from texi.preprocessing import LabelEncoder
from texi.pytorch.dataset import Dataset
from texi.pytorch.transformers.spert.sampler import SpERTSampler

if TYPE_CHECKING:
    from transformers import BertTokenizer, BertTokenizerFast


def create_span_mask(
    starts: List[int],
    ends: List[int],
    length: int,
    dtype: torch.dtype = torch.int64,
    device: torch.device = "cpu",
) -> torch.Tensor:
    if len(starts) != len(ends):
        raise ValueError(
            f"`start` and `end` should have same lengths: {len(starts)} != {len(ends)}"
        )

    if len(starts) == 0:
        return torch.zeros((0, length), dtype=dtype, device=device)

    start = torch.tensor(starts, dtype=dtype, device=device)
    end = torch.tensor(ends, dtype=dtype, device=device)
    mask = torch.arange(length, dtype=dtype, device=device).unsqueeze(dim=-1)
    mask = (start <= mask) & (mask < end)
    mask = mask.transpose(0, 1).type_as(start)

    return mask


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
                    "token_span": [entity["start"], entity["end"]],
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
        entity_masks, entity_labels, entity_token_spans = [], [], []
        max_length, max_entities = 0, 0
        for i, entities in enumerate(collated["entities"]):
            assert len(entities) > 0, "There must be at least 1 negative entity."

            entities = collate(entities)

            num_tokens = len(collated["output"][i]["input_ids"])
            masks = create_span_mask(entities["start"], entities["end"], num_tokens)
            entity_masks += [masks]

            labels = torch.tensor(entities["label"], dtype=torch.int64)
            entity_labels += [labels]

            token_spans = torch.tensor(entities["token_span"], dtype=torch.int64)
            entity_token_spans += [token_spans]

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
        entity_token_spans = [
            torch.nn.functional.pad(x, [0, 0, 0, max_entities - len(x)])
            for x in entity_token_spans
        ]

        entity_masks = torch.stack(entity_masks)
        entity_labels = torch.stack(entity_labels)
        entity_sample_masks = (entity_masks.sum(dim=-1) > 0).long()
        entity_token_spans = torch.stack(entity_token_spans)

        return {
            "entity_masks": entity_masks,
            "entity_labels": entity_labels,
            "entity_sample_masks": entity_sample_masks,
            "entity_token_spans": entity_token_spans,
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
        relation_sample_masks = [
            torch.ones(len(x), dtype=torch.int64) for x in relation_labels
        ]

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
        relation_sample_masks = [
            torch.nn.functional.pad(x, [0, max_relations - x.size(0)])
            for x in relation_sample_masks
        ]

        relation_args = torch.stack(relation_args)
        relation_context_masks = torch.stack(relation_context_masks)
        relation_labels = torch.stack(relation_labels)
        relation_sample_masks = torch.stack(relation_sample_masks)

        return {
            "relations": relation_args,
            "relation_context_masks": relation_context_masks,
            "relation_labels": relation_labels,
            "relation_sample_masks": relation_sample_masks,
        }

    def collate(self, batch: Dict) -> Dict:
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
