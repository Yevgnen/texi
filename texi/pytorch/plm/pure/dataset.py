# -*- coding: utf-8 -*-

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Optional, Union

import torch

from texi.preprocessing import LabelEncoder
from texi.pytorch.dataset import Dataset

if TYPE_CHECKING:
    from transformers import BertTokenizer, BertTokenizerFast


class PureEntityDataset(Dataset):
    def __init__(
        self,
        examples: Iterable[dict],
        entity_label_encoder: LabelEncoder,
        tokenizer: Union[BertTokenizer, BertTokenizerFast] = None,
        train: bool = False,
        eager: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(
            examples, tokenizer=tokenizer, train=train, eager=eager, device=device
        )
        self.entity_label_encoder = entity_label_encoder

    def collate_train(self):
        return

    def collate_eval(self):
        return


class PureRelationDataset(Dataset):
    ...
