# -*- coding: utf-8 -*-

from __future__ import annotations

import abc
import itertools
from collections.abc import Sequence
from typing import Any, Optional

import torch
from carton.collections import collate
from transformers.tokenization_utils import PreTrainedTokenizer

from texi.preprocessing import LabelEncoder
from texi.pytorch.dataset.collator import Collator
from texi.utils import ModeKeys


class PreTrainedCollator(Collator, metaclass=abc.ABCMeta):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        label_encoder: Optional[LabelEncoder] = None,
        mode: ModeKeys = ModeKeys.TRAIN,
    ) -> None:
        super().__init__(mode=mode)
        self.tokenizer = tokenizer
        if label_encoder is None:
            label_encoder = LabelEncoder()
        self.label_encoder = label_encoder

    def collate_fn(self, batch: Sequence) -> Any:
        return self._collate(batch)


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
        tokenizer: PreTrainedTokenizer,
        label_encoder: Optional[LabelEncoder] = None,
        mode: ModeKeys = ModeKeys.TRAIN,
    ) -> None:
        super().__init__(tokenizer, label_encoder=label_encoder, mode=mode)

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
