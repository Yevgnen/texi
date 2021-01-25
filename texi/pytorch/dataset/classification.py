# -*- coding: utf-8 -*-

import torch
from felis.collections import collate
from torchnlp.encoders.text import stack_and_pad_tensors
from torchnlp.utils import collate_tensors, identity

from texi.pytorch.dataset import Dataset


class TextDataset(Dataset):
    def encode(self, example):
        return {
            "text": self.tokenize(example["text"]),
            "label": self.label_encoder.encode(example["label"]),
        }

    def collate(self, batch):
        batch = self.encode_batch(batch)

        batch = collate_tensors(batch, identity)
        texts, text_lengths = stack_and_pad_tensors(batch["text"])
        labels = torch.stack(batch["label"])

        x = {"text": texts, "length": text_lengths}
        y = labels

        return x, y
