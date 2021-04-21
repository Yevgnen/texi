# -*- coding: utf-8 -*-

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Dict, Iterable, Union

import torch
from carton.collections import collate

from texi.preprocessing import LabelEncoder
from texi.pytorch.dataset import Dataset
from texi.pytorch.masking import create_span_mask
from texi.pytorch.transformers.spert.sampler import SpERTSampler

if TYPE_CHECKING:
    from transformers import BertTokenizer, BertTokenizerFast


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

        return encoded_entities

    def _encode_relations(self, relations):
        return [
            {
                "head": x["head"],
                "tail": x["tail"],
                "label": self.relation_label_encoder.encode_label(x["type"]),
            }
            for x in relations
        ]

    def encode_example(self, example, entities, relations):
        # Encode tokens.
        tokens = (
            [self.tokenizer.cls_token] + example["tokens"] + [self.tokenizer.sep_token]
        )
        output = self.tokenizer(tokens, add_special_tokens=False)

        # Encode entities.
        encoded_entities = self._encode_entities(entities, output["input_ids"])

        # Encode relations.
        encoded_relations = self._encode_relations(relations)

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
            mask = create_span_mask(entities["start"], entities["end"], num_tokens)
            entity_masks += [mask]

            label = torch.tensor(entities["label"], dtype=torch.int64)
            entity_labels += [label]

            token_span = torch.tensor(entities["token_span"], dtype=torch.int64)
            entity_token_spans += [token_span]

            max_entities = max(max_entities, mask.size(0))
            max_length = max(max_length, mask.size(1))

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

        entity_mask = torch.stack(entity_masks)
        entity_label = torch.stack(entity_labels)
        entity_sample_mask = (entity_mask.sum(dim=-1) > 0).long()
        entity_token_span = torch.stack(entity_token_spans)

        return {
            "entity_mask": entity_mask,
            "entity_label": entity_label,
            "entity_sample_mask": entity_sample_mask,
            "entity_token_span": entity_token_span,
        }

    def _collate_relations(self, collated):
        def _create_context_mask(heads, tails, entity_spans, length):
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
                head = torch.tensor(relations["head"], dtype=torch.int64)
                tail = torch.tensor(relations["tail"], dtype=torch.int64)
                entity_spans = {
                    i: (x["start"], x["end"]) for i, x in enumerate(entities)
                }
                mask = _create_context_mask(
                    relations["head"], relations["tail"], entity_spans, num_tokens
                )
                arg = torch.stack([head, tail], dim=1)
                label = torch.nn.functional.one_hot(
                    torch.tensor(relations["label"]), len(self.relation_label_encoder)
                )
            else:
                arg = torch.zeros((0, 2), dtype=torch.int64)
                mask = torch.zeros((0, num_tokens), dtype=torch.int64)
                label = torch.zeros(
                    (0, len(self.relation_label_encoder)), dtype=torch.int64
                )

            relation_args += [arg]
            relation_context_masks += [mask]
            relation_labels += [label]

            max_relations = max(max_relations, arg.size(0))
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

        relation_arg = torch.stack(relation_args)
        relation_context_mask = torch.stack(relation_context_masks)
        relation_label = torch.stack(relation_labels)
        relation_sample_mask = torch.stack(relation_sample_masks)

        return {
            "relation": relation_arg,
            "relation_context_mask": relation_context_mask,
            "relation_label": relation_label,
            "relation_sample_mask": relation_sample_mask,
        }

    def collate(self, batch: Dict) -> Dict[str, torch.Tensor]:
        encoded_batch = [self.encode(x) for x in batch]
        collated = collate(encoded_batch)

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
