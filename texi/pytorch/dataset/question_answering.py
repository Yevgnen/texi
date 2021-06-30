# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch
from carton.collections import collate
from torchnlp.encoders.text import stack_and_pad_tensors
from torchnlp.utils import collate_tensors, identity

from texi.pytorch.dataset.collator import Collator

if TYPE_CHECKING:
    from texi.datasets.dataset import Dataset


class QuestionAnsweringCollator(Collator):
    def __init__(self, dataset: Dataset, tokenizer: Callable) -> None:
        super().__init__(dataset)
        self.tokenizer = tokenizer

    def encode(self, example):
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
            "context": self.tokenizer(example["context"]),
            "question": self.tokenizer(example["question"]),
            "answers": _encode_answers(example["context"], example["answers"]),
        }

    def collate_train(self, batch):
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
