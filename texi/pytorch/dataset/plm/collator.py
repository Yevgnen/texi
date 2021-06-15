# -*- coding: utf-8 -*-

from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Optional

from transformers.tokenization_utils import PreTrainedTokenizer

from texi.preprocessing import LabelEncoder
from texi.pytorch.dataset.collator import Collator

if TYPE_CHECKING:
    from texi.datasets.dataset import Dataset


class PreTrainedCollator(Collator, metaclass=abc.ABCMeta):
    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        label_encoder: Optional[LabelEncoder] = None,
    ) -> None:
        super().__init__(dataset)
        self.tokenizer = tokenizer
        if label_encoder is None:
            label_encoder = LabelEncoder()
        self.label_encoder = label_encoder

    def collate_fn(self, batch: Sequence) -> Any:
        return self._collate(batch)
