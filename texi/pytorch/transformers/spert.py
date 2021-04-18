# -*- coding: utf-8 -*-

import itertools
import random
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch

from texi.preprocessing import LabelEncoder
from texi.pytorch.dataset import Dataset


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
        tokenizer: Optional[Any] = None,
        train: bool = False,
    ):
        super().__init__(examples, tokenizer=tokenizer, train=train)
        self.negative_sampler = negative_sampler
        self.entity_label_encoder = entity_label_encoder
        self.relation_label_encoder = relation_label_encoder

    def _encode_entities(self, entities, tokens):
        offset = 1  # Added [CLS] token.
        offsets = []
        for token in tokens:
            offsets += [offset]
            offset += len(token)

        entity_indices = {}
        encoded_entities = []
        for i, entity in enumerate(entities):
            start = offsets[entity["start"]]
            end = offsets[entity["end"] - 1] + len(tokens[entity["end"] - 1])
            encoded_entities += [
                {
                    "start": start,
                    "end": end,
                    "type": self.entity_label_encoder.encode_label(entity["type"]),
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
                    "type": self.relation_label_encoder.encode_label(rel["type"]),
                }
            ]

        return encoded_relations

    def encode_example(self, example, entities, relations):
        output = self.tokenizer(example["text"], add_special_tokens=False)

        encoded_entities, entity_indices = self._encode_entities(
            entities, output["input_ids"]
        )
        encoded_relations = self._encode_relations(relations, entity_indices)

        output = {k: list(itertools.chain.from_iterable(v)) for k, v in output.items()}
        return {
            "output": output,
            "entities": encoded_entities,
            "relations": encoded_relations,
        }
