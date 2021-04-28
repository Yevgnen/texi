# -*- coding: utf-8 -*-

import torch
from torchnlp.encoders.text import stack_and_pad_tensors
from torchnlp.utils import collate_tensors, identity

from texi.preprocessing import LabelEncoder
from texi.pytorch.dataset import Dataset


class SequenceLabelingDataset(Dataset):
    def __init__(self, *args, label_encoder: LabelEncoder, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_encoder = label_encoder

    def encode(self, example):
        return {
            "tokens": torch.cat([self.tokenizer.encode(x) for x in example["tokens"]]),
            "labels": self.label_encoder.encode(example["labels"]),
        }

    def collate(self, batch):
        batch = self.encode_batch(batch)

        batch = collate_tensors(batch, identity)
        tokens, token_lengths = stack_and_pad_tensors(batch["tokens"])
        labels, label_lengths = stack_and_pad_tensors(batch["labels"])
        assert all(token_lengths == label_lengths)

        x = {
            "token": tokens,
            "length": token_lengths,
        }
        y = labels

        return x, y
