# -*- coding: utf-8 -*-

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from carton.collections import collate

from texi.pytorch.dataset import TextPairDataset as _TextPairDataset
from texi.pytorch.plm.dataset.collator import PreTrainedCollator


class TextClassificationCollator(PreTrainedCollator):
    def collate_train(self, batch: Sequence) -> Any:
        collated = collate(batch)

        x = self.tokenizer(
            collated["text"],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        y = torch.tensor(collated["label"], dtype=torch.int64)

        return x, y


class TextPairDataset(_TextPairDataset):
    def collate(self, batch):
        collated = collate(batch)

        text_pairs = list(zip(collated["sentence1"], collated["sentence2"]))
        x = self.tokenizer(
            text_pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        y = torch.tensor(collated["label"], dtype=torch.int64)

        return x, y
