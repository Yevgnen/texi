# -*- coding: utf-8 -*-

import torch
from felis.collections import collate

from texi.pytorch.dataset import TextDataset as _TextDataset
from texi.pytorch.dataset import TextPairDataset as _TextPairDataset


class TextDataset(_TextDataset):
    def collate(self, batch):
        collated = collate(batch)

        x = self.tokenizer(
            list(collated["text"]), padding=True, truncation=True, return_tensors="pt"
        )
        y = torch.tensor(collated["label"], dtype=torch.int64)

        return x, y


class TextPairDataset(_TextPairDataset):
    def collate(self, batch):
        collated = collate(batch)

        text_pairs = list(zip(collated["query"], collated["doc"]))
        x = self.tokenizer(
            text_pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        y = torch.tensor(collated["label"], dtype=torch.int64)

        return x, y
