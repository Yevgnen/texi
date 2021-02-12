# -*- coding: utf-8 -*-


import torch
from carton.collections import collate
from torchnlp.encoders.text import stack_and_pad_tensors
from torchnlp.utils import collate_tensors, identity

from texi.pytorch.dataset import Dataset


class QuestionAnsweringDataset(Dataset):
    def _encode(self, example):
        def _encode_answers(context, answers):
            encoded = {
                "start": torch.zeros(len(context)),
                "end": torch.zeros(len(context)),
            }
            indices = torch.arange(len(answers), dtype=torch.float32) + 1
            encoded["start"][[x["start"] for x in answers]] = indices
            encoded["end"][[x["end"] - 1 for x in answers]] = indices

            return encoded

        return {
            "context": self.tokenize(example["context"]),
            "question": self.tokenize(example["question"]),
            "answers": _encode_answers(example["context"], example["answers"]),
        }

    def _collate(self, batch):
        batch = self.encode(batch)

        batch = collate(batch)
        contexts, context_lengths = stack_and_pad_tensors(batch["context"])
        questions, question_lengths = stack_and_pad_tensors(batch["question"])

        answers = collate_tensors(batch["answers"], identity)
        starts, _ = stack_and_pad_tensors(answers["start"])
        ends, _ = stack_and_pad_tensors(answers["end"])

        x = {
            "question": questions,
            "query_length": question_lengths,
            "context": contexts,
            "context_length": context_lengths,
        }
        y = torch.stack([starts, ends]).type(torch.int64)

        return x, y
