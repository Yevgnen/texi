# -*- coding: utf-8 -*-

import torch
from torchnlp.encoders.text import stack_and_pad_tensors
from torchnlp.utils import collate_tensors, identity

from texi.pytorch.dataset import Dataset
from texi.pytorch.utils import length_to_mask


class SequenceLabelingDataset(Dataset):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("label_encoder", "tags")
        super().__init__(*args, **kwargs)

    def encode(self, example):
        return {
            "tokens": torch.cat([self.tokenizer.encode(x) for x in example["tokens"]]),
            "tags": self.label_encoder.encode(example["tags"]),
        }

    def collate(self, batch):
        batch = self.encode_batch(batch)

        batch = collate_tensors(batch, identity)
        tokens, token_lengths = stack_and_pad_tensors(batch["tokens"])
        tags, tag_lengths = stack_and_pad_tensors(batch["tags"])
        assert all(token_lengths == tag_lengths)

        x = {
            "token": tokens,
            "length": token_lengths,
            "tag_mask": length_to_mask(token_lengths, batch_first=True),
        }
        y = tags

        return x, y
