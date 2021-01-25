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
            "query": self.tokenize(example["query"]),
            "doc": self.tokenize(example["doc"]),
            "label": self.label_encoder.encode(example["label"]),
        }

    def collate(self, batch):
        batch = self.encode_batch(batch)

        batch = collate_tensors(batch, identity)
        queries, query_lengths = stack_and_pad_tensors(batch["query"])
        docs, doc_lengths = stack_and_pad_tensors(batch["doc"])

        x = {
            "query": queries,
            "doc": docs,
            "query_length": query_lengths,
            "doc_length": doc_lengths,
        }
        y = torch.tensor(batch["label"], dtype=torch.int64)

        return x, y
