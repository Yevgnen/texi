# -*- coding: utf-8 -*-

from __future__ import annotations

from collections.abc import Sequence

import torch
from carton.collections import collate

from texi.pytorch.plm.dataset.collator import PreTrainedCollator


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
