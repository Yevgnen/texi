# -*- coding: utf-8 -*-

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Optional, TypeVar, cast

import ignite.distributed as idist
from torch.utils.data import DataLoader

from texi.pytorch.utils import get_sampler

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

    def get_dataloader(
        self,
        batch_size: int,
        drop_last: bool = False,
        sort_key: Callable = lambda x: x,
        **kwargs
    ) -> DataLoader:
        sampler = get_sampler(
            cast(Sequence, self.dataset.examples),
            self.dataset.is_train(),
            batch_size,
            drop_last=drop_last,
            sort_key=lambda index: sort_key(self.dataset[index]),
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
