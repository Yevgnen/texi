# -*- coding: utf-8 -*-

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import torch
from carton.collections import collate
from torchnlp.encoders.text import stack_and_pad_tensors
from torchnlp.utils import collate_tensors, identity

from texi.preprocessing import LabelEncoder
from texi.utils import ModeKeys, PhaseMixin


class Collator(PhaseMixin, metaclass=abc.ABCMeta):
    T = TypeVar("T", bound="Collator")

    def __init__(self, mode: ModeKeys = ModeKeys.TRAIN) -> None:
        self.mode = mode

    def __call__(self, batch: Sequence) -> Any:
        return self.collate_fn(batch)

    def encode(self, example) -> Any:  # pylint: disable=no-self-use
        return example

    def encode_batch(self, batch: Sequence) -> list:
        return list(map(self.encode, batch))

    def collate_train(self, batch: Sequence) -> Any:
        raise NotImplementedError()

    def collate_eval(self, batch: Sequence) -> Any:
        return self.collate_train(batch)

    def collate_predict(self, batch: Sequence) -> Any:
        raise NotImplementedError()

    def _get_collate_fn(self):
        return {
            ModeKeys.TRAIN: self.collate_train,
            ModeKeys.EVAL: self.collate_eval,
            ModeKeys.PREDICT: self.collate_predict,
        }[self.mode]

    def _collate(self, encoded_batch):
        return self._get_collate_fn()(encoded_batch)

    def collate_fn(self, batch: Sequence) -> Any:
        encoded_batch = self.encode_batch(batch)
        collated = self._collate(encoded_batch)

        return collated


class TextClassificationCollator(Collator):
    def __init__(
        self,
        tokenizer: Callable,
        label_encoder: LabelEncoder,
        mode: ModeKeys = ModeKeys.TRAIN,
    ) -> None:
        super().__init__(mode=mode)
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
        tokenizer: Callable,
        label_encoder: LabelEncoder,
        mode: ModeKeys = ModeKeys.TRAIN,
    ) -> None:
        super().__init__(mode=mode)
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


class QuestionAnsweringCollator(Collator):
    def __init__(self, tokenizer: Callable, mode: ModeKeys = ModeKeys.TRAIN) -> None:
        super().__init__(mode=mode)
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
