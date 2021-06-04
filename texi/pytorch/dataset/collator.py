# -*- coding: utf-8 -*-

from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from texi.datasets.dataset import Dataset


class Collator(object, metaclass=abc.ABCMeta):
    T = TypeVar("T", bound="Collator")

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

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

    def collate_fn(self, batch: Sequence) -> Any:
        encoded = self.encode_batch(batch)

        fn = self.collate_train if self.dataset.is_train() else self.collate_eval
        collated = fn(encoded)

        return collated
