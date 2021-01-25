# -*- coding: utf-8 -*-

import torch
from felis.collections import collate

from texi.pytorch.dataset import TextDataset as _TextDataset


class TextDataset(_TextDataset):
    def collate(self, batch):
        collated = collate(batch)

        x = self.tokenizer(
            list(collated["text"]), padding=True, truncation=True, return_tensors="pt"
        )
        y = torch.tensor(collated["label"], dtype=torch.int64)

        return x, y
