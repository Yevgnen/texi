# -*- coding: utf-8 -*-

import torch
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


class TextPairDataset(Dataset):
    def encode(self, example):
        return {
            "sentence1": self.tokenize(example["sentence1"]),
            "sentence2": self.tokenize(example["sentence2"]),
            "label": self.label_encoder.encode(example["label"]),
        }

    def collate(self, batch):
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
