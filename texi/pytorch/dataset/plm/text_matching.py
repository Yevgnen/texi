# -*- coding: utf-8 -*-

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional

import torch
from carton.collections import collate
from transformers.tokenization_utils import PreTrainedTokenizer

from texi.preprocessing import LabelEncoder
from texi.pytorch.dataset.plm.collator import PreTrainedCollator

if TYPE_CHECKING:
    from texi.datasets.dataset import Dataset


class TextClassificationCollator(PreTrainedCollator):
    def collate_train(
        self, batch: Sequence
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        collated = collate(batch)

        x = self.tokenizer(
            collated["text"],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        y = self.label_encoder.encode(collated["label"], return_tensors="pt")

        return x, y


class TextMatchingCollator(PreTrainedCollator):
    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        label_encoder: Optional[LabelEncoder] = None,
    ) -> None:
        super().__init__(dataset, tokenizer, label_encoder=label_encoder)

    def collate_train(
        self, batch: Sequence
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        # {
        #     "texts": ["sentence1", "sentence2", ...]
        #     "label": 1
        # }

        collated = collate(batch)

        batch_size = len(batch)
        texts = list(itertools.chain.from_iterable(zip(*collated["texts"])))
        assert (
            len(texts) % batch_size == 0
        ), 'All exmaples should have size of "texts" fields'

        x = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        chunks = len(texts) // batch_size

        def _stack(t):
            return torch.stack(t.chunk(chunks, dim=0), dim=0)

        x = {k: _stack(v) for k, v in x.items()}
        y = self.label_encoder.encode(collated["label"], return_tensors="pt")

        return x, y
