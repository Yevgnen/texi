# -*- coding: utf-8 -*-

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

import torch
from carton.collections import collate, flatten_dict

from texi.apps.ner.utils import NerExample, describe_examples
from texi.preprocessing import LabelEncoder
from texi.pytorch.dataset import Dataset
from texi.pytorch.dataset.dataset import EagerEncodeMixin
from texi.pytorch.masking import length_to_mask
from texi.pytorch.utils import pad_stack_1d, pad_stack_2d
from texi.utils import ModeKeys

if TYPE_CHECKING:
    from transformers import BertTokenizerFast


class Mrc4NerDataset(EagerEncodeMixin, Dataset):
    def __init__(
        self,
        examples: Iterable[NerExample],
        entity_label_encoder: LabelEncoder,
        max_entity_size: int,
        queries: Union[Mapping[str, str]],
        tokenizer: BertTokenizerFast = None,
        mode: ModeKeys = ModeKeys.TRAIN,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(examples, tokenizer=tokenizer, mode=mode, device=device)

        self.entity_label_encoder = entity_label_encoder
        self.max_entity_size = max_entity_size
        self.queries = queries
        self.tokenized_queries = {
            key: self.tokenizer.tokenize(value) for key, value in self.queries.items()
        }

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        type_stats = flatten_dict(describe_examples(self.examples))
        info.update(type_stats)

        return info

    def encode(self, example: NerExample) -> list[dict]:
        def _encode_by_type(type_, entities):
            query_tokens = self.tokenized_queries[type_]
            tokenizer_output = self.tokenizer(
                query_tokens,
                example["tokens"],
                is_split_into_words=True,
                return_offsets_mapping=True,
            )

            length = len(tokenizer_output["input_ids"])
            starts, ends = [0] * length, [0] * length
            entity_iter = iter(entities)
            entity = next(entity_iter, None)
            query_length = len(query_tokens)
            i = 0
            index_mapping = {}

            for j, (offset_start, _) in enumerate(tokenizer_output["offset_mapping"]):
                if offset_start == 0:
                    i += 1
                    if entity is not None:
                        if i - query_length - 2 == entity["start"]:
                            starts[i] = 1
                            entity = next(entity_iter, None)
                        elif i - query_length - 2 == entity["end"]:
                            ends[i] = 1
                            entity = next(entity_iter, None)
                    index_mapping[i] = j

            start = torch.tensor(starts, dtype=torch.int64)
            end = torch.tensor(ends, dtype=torch.int64)
            span_indices = [
                [index_mapping[i], index_mapping[i + j]]
                for i in index_mapping
                for j in range(self.max_entity_size)
                if i + j < length - 1
            ]
            span_index = torch.tensor(span_indices, dtype=torch.int64)

            return {
                "input_ids": torch.tensor(
                    tokenizer_output["input_ids"], dtype=torch.int64
                ),
                "attention_mask": torch.tensor(
                    tokenizer_output["attention_mask"], dtype=torch.int64
                ),
                "token_type_ids": torch.tensor(
                    tokenizer_output["token_type_ids"], dtype=torch.int64
                ),
                "start": start,
                "end": end,
                "span_index": span_index,
            }

        def _sort_key(x):
            return x["type"]

        encoded = []
        for key, entities in itertools.groupby(
            sorted(example["entities"], key=_sort_key), key=_sort_key
        ):
            encoded += [_encode_by_type(key, entities)]

        return encoded

    def collate_train(
        self, batch: Sequence[Sequence[Mapping]]
    ) -> dict[str, torch.Tensor]:
        collated = collate(itertools.chain.from_iterable(batch))
        max_length = max(len(x) for x in collated["input_ids"])

        input_ids = pad_stack_1d(collated["input_ids"], max_length)
        attention_mask = pad_stack_1d(collated["attention_mask"], max_length)
        token_type_ids = pad_stack_1d(collated["token_type_ids"], max_length)
        start = pad_stack_1d(collated["start"], max_length)
        end = pad_stack_1d(collated["end"], max_length)

        num_spans = torch.tensor([len(x) for x in collated["span_index"]])
        span_mask = length_to_mask(num_spans, batch_first=True)
        span_index = pad_stack_2d(collated["span_index"], max(num_spans), 2)

        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "start": start,
            "end": end,
            "span_index": span_index,
            "span_mask": span_mask,
        }

        return output
