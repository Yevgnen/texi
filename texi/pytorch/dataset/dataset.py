# -*- coding: utf-8 -*-

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import torch
from torch.utils.data import DataLoader

from texi.datasets import Dataset as BaseDataset
from texi.pytorch.utils import get_sampler

Batch = Union[Tuple[Dict[str, torch.Tensor], torch.Tensor], Dict[str, torch.Tensor]]
Texts = Union[Iterable[str], str]


class Dataset(BaseDataset):
    T = TypeVar("T", bound="Dataset")

    def __init__(
        self,
        examples: Union[Iterable[Dict], Callable[[], Iterable[Dict]]],
        tokenizer: Optional[Any] = None,
        train: bool = False,
    ) -> None:
        super().__init__(examples)

        if tokenizer is None:
            raise ValueError("`tokenizer` must not be None")
        self.tokenizer = tokenizer
        self.label_encoder = None

        self.is_train = train

    def train(self) -> None:
        self.is_train = True

    def eval(self) -> None:
        self.is_train = False

    def encode(self, example: Mapping) -> Dict:
        raise NotImplementedError()

    def collate(self, batch: Batch) -> Any:
        raise NotImplementedError()

    def encode_batch(self, batch: Batch) -> List[Dict]:
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
            self.examples,
            self.is_train,
            batch_size,
            drop_last=drop_last,
            sort_key=sort_key,
        )

        collate_fn = kwargs.pop("collate_fn", self.collate)
        dataloader = DataLoader(
            self, batch_sampler=sampler, collate_fn=collate_fn, **kwargs
        )

        return dataloader

    @staticmethod
    def get_dataloaders(
        iterators: Dict[str, T],
        train_batch_size: Optional[int] = None,
        eval_batch_size: Optional[int] = None,
        batch_size: Optional[int] = None,
        drop_last: bool = False,
        sort_key: Callable = lambda x: x,
        **kwargs
    ) -> Dict[str, DataLoader]:
        # 1. Train dataset has individual batch size.
        # 2. `drop_last` will alwarys be False for val and test datasets.
        # 3. `sort_key` is passed only in train dataset.

        batch_sizes = {
            "train": batch_size if train_batch_size is None else train_batch_size,
            "val": batch_size if eval_batch_size is None else eval_batch_size,
            "test": batch_size if eval_batch_size is None else eval_batch_size,
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

    @classmethod
    def get_iterators(
        cls, datasets: Sequence[Iterable], *args, **kwargs
    ) -> Dict[str, T]:
        assert len(datasets) in {1, 2, 3}

        train, *others = datasets
        kwargs["train"] = True
        train_iter = cls(train, *args, **kwargs)

        kwargs["train"] = False
        kwargs["tokenizer"] = train_iter.tokenizer
        kwargs["label_encoder"] = train_iter.label_encoder
        others = tuple(cls(other, *args, **kwargs) for other in others)

        if not others:
            others = (None,) * 2
        elif len(others) == 1:
            others = (None,) + others
        iterators = dict(zip(["val", "test"], others))
        iterators["train"] = train_iter

        return iterators
