# -*- coding: utf-8 -*-

from __future__ import annotations

from collections.abc import Callable

import torch
from torchnlp.encoders.text import stack_and_pad_tensors
from torchnlp.utils import collate_tensors, identity

from texi.datasets import Dataset
from texi.preprocessing import LabelEncoder
from texi.pytorch.dataset.collator import Collator


class TextClassificationCollator(Collator):
    def __init__(
        self,
        dataset: Dataset,
        tokenizer: Callable,
        label_encoder: LabelEncoder,
    ) -> None:
        super().__init__(dataset)
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder

    def encode(self, example):
        return {
            "text": self.tokenizer(example["text"]),
            "label": self.label_encoder.encode_label(example["label"]),
        }

    def collate_train(self, batch):
        batch = self.encode_batch(batch)

        batch = collate_tensors(batch, identity)
        text, length = stack_and_pad_tensors(batch["text"])
        label = torch.stack(batch["label"])

        x = {
            "text": text,
            "length": length,
        }
        y = label

        return x, y


class TextMatchingCollator(Collator):
    def __init__(
        self,
        dataset: Dataset,
        tokenizer: Callable,
        label_encoder: LabelEncoder,
    ) -> None:
        super().__init__(dataset)
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder

    def encode(self, example):
        return {
            "sentence1": self.tokenizer(example["sentence1"]),
            "sentence2": self.tokenizer(example["sentence2"]),
            "label": self.label_encoder.encode(example["label"]),
        }

    def collate_train(self, batch):
        batch = self.encode_batch(batch)

        batch = collate_tensors(batch, identity)
        sentence1, length1 = stack_and_pad_tensors(batch["sentence1"])
        sentence2, length2 = stack_and_pad_tensors(batch["sentence2"])

        x = {
            "sentence1": sentence1,
            "sentence2": sentence2,
            "length1": length1,
            "length2": length2,
        }
        y = torch.tensor(batch["label"], dtype=torch.int64)

        return x, y
