# -*- coding: utf-8 -*-

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Iterable, Optional, TypeVar, Union, cast

import ignite.distributed as idist
import torch
from carton.collections import collate
from torch.utils.data import DataLoader

from texi.datasets import Dataset as BaseDataset
from texi.pytorch.utils import get_sampler

Texts = Union[Iterable[str], str]


class Dataset(torch.utils.data.Dataset, BaseDataset):
    T = TypeVar("T", bound="Dataset")

    def __init__(
        self,
        examples: Union[Iterable, Callable],
        tokenizer: Optional[Any] = None,
        train: bool = False,
        eager: bool = True,
    ) -> None:
        super().__init__(examples)

        if tokenizer is None:
            raise ValueError("`tokenizer` must not be None")
        self.tokenizer = tokenizer
        self.label_encoder = None

        self.is_train = train

        self.eager = eager
        self._encoded_examples = None

    def __getitem__(self, key):
        if self._encoded_examples is None:
            self._eager_encode_maybe()

        return self._encoded_examples[key]

    def _eager_encode_maybe(self):
        if self.eager:
            self._encoded_examples = self.encode_batch(cast(list, self.examples))
        else:
            self._encoded_examples = cast(list, self.examples)

    def train(self) -> None:
        self.is_train = True
        self._eager_encode_maybe()

    def eval(self) -> None:
        self.is_train = False
        self._eager_encode_maybe()

    def encode(self, example: Any) -> Any:
        raise NotImplementedError()

    def _collate_internal(self, batch):
        raise NotImplementedError()

    def collate_eager(self, batch: Sequence) -> Any:
        # The differences between this method and `collate` is: this
        # method assumes `batch` is already encoded. Hopefully this will
        # speed up the collating process of `DataLoader`.

        if not self.eager:
            raise ValueError("`collate_eager` must only be called when `eager` = True")

        collated = self._collate_internal(collate(batch))

        return collated

    def collate(self, batch: Sequence) -> Any:
        if self.eager:
            raise ValueError("`collate` must only be called when `eager` = False")

        encoded = self.encode_batch(batch)
        collated = self._collate_internal(collate(encoded))

        return collated

    def collate_fn(self, batch: Sequence) -> Any:
        if self.eager:
            return self.collate_eager(batch)

        return self.collate(batch)

    def encode_batch(self, batch: Sequence) -> list:
        return [*map(self.encode, batch)]

    def tokenize(self, text: Texts) -> torch.Tensor:
        if callable(self.tokenizer):
            return self.tokenizer(text)

        return self.tokenizer.encode(text)

    def get_dataloader(
        self,
        batch_size: int,
        drop_last: bool = False,
        sort_key: Callable = lambda x: x,
        **kwargs
    ) -> DataLoader:
        sampler = get_sampler(
            cast(Sequence, self.examples),
            self.is_train,
            batch_size,
            drop_last=drop_last,
            sort_key=lambda index: sort_key(self[index]),
        )

        collate_fn = kwargs.pop("collate_fn", self.collate_fn)
        dataloader = idist.auto_dataloader(
            self, batch_sampler=sampler, collate_fn=collate_fn, **kwargs
        )  # type: DataLoader[Dataset]

        return dataloader

    @staticmethod
    def get_dataloaders(
        iterators: dict[str, T],
        train_batch_size: Optional[int] = None,
        eval_batch_size: Optional[int] = None,
        batch_size: Optional[int] = None,
        drop_last: bool = False,
        sort_key: Callable = lambda x: x,
        **kwargs
    ) -> dict[str, DataLoader]:
        # 1. Train dataset has individual batch size.
        # 2. `drop_last` will alwarys be False for val and test datasets.
        # 3. `sort_key` is passed only in train dataset.

        if (train_batch_size is None or eval_batch_size is None) and batch_size is None:
            raise ValueError(
                "`batch_size` must not be None"
                " if `train_batch_size` or `eval_batch_size` is None"
            )

        if train_batch_size is None:
            train_batch_size = cast(int, batch_size)

        if eval_batch_size is None:
            eval_batch_size = cast(int, batch_size)

        batch_sizes = {
            "train": train_batch_size,
            "val": eval_batch_size,
            "test": eval_batch_size,
        }

        loaders = {}
        for mode, iterator in iterators.items():
            if mode == "train":
                loader = iterator.get_dataloader(
                    batch_sizes[mode], drop_last=drop_last, sort_key=sort_key, **kwargs
                )
            else:
                loader = iterator.get_dataloader(
                    batch_sizes[mode], drop_last=False, **kwargs
                )

            loaders[mode] = loader

        return loaders
