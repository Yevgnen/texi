# -*- coding: utf-8 -*-

from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Optional

import torch
from ignite.utils import convert_tensor
from transformers.tokenization_utils import PreTrainedTokenizer

from texi.pytorch.dataset.collator import Collator

if TYPE_CHECKING:
    from texi.datasets.dataset import Dataset


class PreTrainedCollator(Collator, metaclass=abc.ABCMeta):
    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(dataset, device=device)
        self.tokenizer = tokenizer

    def collate_fn(self, batch: Sequence) -> Any:
        fn = self.collate_train if self.dataset.is_train() else self.collate_eval

        collated = fn(batch)

        if self.device is not None:
            collated = convert_tensor(collated, device=self.device, non_blocking=True)

        return collated
