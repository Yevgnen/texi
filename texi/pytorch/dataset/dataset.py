# -*- coding: utf-8 -*-

import abc
from typing import (
    Any,
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

from texi.pytorch.utils import get_sampler

T = TypeVar("T", bound="Dataset")
S = TypeVar("S")
Batch = Tuple[Dict[str, torch.Tensor], torch.Tensor]
Texts = Union[Iterable[str], str]


class Dataset(metaclass=abc.ABCMeta):
    def __init__(
        self,
        examples: Iterable[Dict],
        tokenizer: Optional[Any] = None,
        train: bool = False,
    ):
        self.examples = list(examples)

        if tokenizer is None:
            raise ValueError("`tokenizer` must not be None")
        self.tokenizer = tokenizer
        self.label_encoder = None

        self.train = train

    def encode(self, example: Mapping) -> Dict:
        raise NotImplementedError()

    def collate(self, batch: Batch) -> Dict:
        raise NotImplementedError()

    def encode_batch(self, batch: List[Mapping]) -> List[Dict]:
        return [*map(self.encode, batch)]

    def tokenize(self, text: Texts) -> torch.Tensor:
        if callable(self.tokenizer):
            return self.tokenizer(text)

        return self.tokenizer.encode(text)

    def get_dataloader(
        self, sampler_kwargs: Optional[Mapping] = None, **kwargs
    ) -> DataLoader:
        if not sampler_kwargs:
            sampler_kwargs = {}
        sampler = get_sampler(self.examples, train=self.train, **sampler_kwargs)

        collate_fn = kwargs.pop("collate_fn", self.collate)
        dataloader = DataLoader(
            self.examples, batch_sampler=sampler, collate_fn=collate_fn, **kwargs
        )

        return dataloader

    @staticmethod
    def get_dataloaders(
        iterators: Dict[str, "Dataset"], *args, **kwargs
    ) -> Dict[str, DataLoader]:
        return {k: v.get_dataloader(*args, **kwargs) for k, v in iterators.items()}

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
