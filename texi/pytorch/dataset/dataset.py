# -*- coding: utf-8 -*-

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Iterable, Optional, TypeVar, Union, cast

import ignite.distributed as idist
import torch
from ignite.utils import convert_tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from texi.datasets import Dataset as BaseDataset
from texi.preprocessing import LabelEncoder
from texi.pytorch.utils import get_sampler
from texi.utils import ModeKeys

Texts = Union[Iterable[str], str]


class Dataset(BaseDataset, TorchDataset):

    T = TypeVar("T", bound="Dataset")

    def __init__(
        self,
        examples: Union[Iterable, Callable],
        tokenizer: Optional[Any] = None,
        mode: ModeKeys = ModeKeys.TRAIN,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(examples)

        if tokenizer is None:
            raise ValueError("`tokenizer` must not be None")
        self.tokenizer = tokenizer
        self.label_encoder = None  # type: Optional[LabelEncoder]

        self.mode = mode
        self.device = device

    def tokenize(self, text: Texts) -> torch.Tensor:
        if callable(self.tokenizer):
            return self.tokenizer(text)

        return self.tokenizer.encode(text)

    def encode(self, example: Any) -> Any:  # pylint: disable=no-self-use
        return example

    def encode_batch(self, batch: Sequence) -> list:
        return [*map(self.encode, batch)]

    def collate_train(self, batch: Sequence) -> Any:
        raise NotImplementedError()

    def collate_eval(self, batch: Sequence) -> Any:
        return self.collate_train(batch)

    def collate_fn(self, batch: Sequence) -> Any:
        encoded = self.encode_batch(batch)

        if self.device is not None:
            encoded = convert_tensor(encoded, device=self.device, non_blocking=True)

        fn = self.collate_train if self.is_train() else self.collate_eval
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
            cast(Sequence, self.examples),
            self.is_train(),
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
