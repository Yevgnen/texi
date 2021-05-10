# -*- coding: utf-8 -*-

from __future__ import annotations

import itertools
import json
import os
from collections.abc import Callable, Iterable
from typing import Any, Optional, Type, TypeVar, Union, cast


class Dataset(object):
    T = TypeVar("T", bound="Dataset")

    def __init__(self, examples: Union[Iterable, Callable]) -> None:
        if callable(examples):
            self.load_examples = examples  # type: Optional[Callable]
            self.examples = None
        else:
            self.examples = list(examples)
            self.load_examples = None

    def _check_loaded(self):
        if self.examples is None:
            raise RuntimeError("Dataset is not loaded, call `.load()` first")

    def load(self) -> T:
        if callable(self.load_examples) and self.examples is None:
            self.examples = list(self.load_examples())

        return self

    def map(self, fn: Callable) -> None:
        self._check_loaded()

        examples = [fn(x) for x in cast(list, self.examples)]
        if examples and isinstance(examples[0], list):
            examples = list(itertools.chain.from_iterable(examples))

        self.examples = examples

    def describe(self) -> dict[str, Any]:
        return {"size": len(self)}

    def __getitem__(self, key):
        self._check_loaded()

        return self.examples[key]

    def __iter__(self):
        if self.examples is not None:
            yield from iter(self.examples)
        elif callable(self.load_examples):
            yield from self.load_examples()

    def __len__(self):
        self._check_loaded()

        return len(self.examples)

    def __repr__(self):
        if self.examples is None:
            return f"{self.__class__.__name__}(Not loaded)"

        return f"{self.__class__.__name__}({len(self)} examples)"

    @classmethod
    def from_json_iter(
        cls: Type[T],
        filename: str,
        format_function: Optional[Callable] = lambda x: x,
        array: bool = False,
    ) -> T:
        def _iter_whole_file():
            with open(filename) as f:
                yield from map(format_function, json.load(f))

        def _iter_multiple_lines():
            with open(filename) as f:
                for line in f:
                    line = line.rstrip()
                    if line:
                        yield format_function(json.loads(line))

        fn = _iter_whole_file if array else _iter_multiple_lines

        return cls(fn)

    @classmethod
    def from_json(cls: Type[T], filename: str) -> T:
        return cls(list(cls.from_json_iter(filename)))


class Datasets(object):
    T = TypeVar("T", bound="Datasets")

    def __init__(
        self,
        train: Optional[Dataset] = None,
        val: Optional[Dataset] = None,
        test: Optional[Dataset] = None,
        dirname: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> None:
        self.train = train
        self.val = val
        self.test = test
        self.dirname = dirname
        self.filename = filename

        self.modes = {"train", "val", "test"}

    def load(self) -> "Datasets":
        for mode in self.modes:
            dataset = getattr(self, mode)
            if dataset is not None:
                dataset.load()

        return self

    def items(self) -> Iterable[tuple[str, Dataset]]:
        for mode in self.modes:
            yield mode, getattr(self, mode)

    def map(self, fn: Callable) -> None:
        for _, dataset in self.items():
            if dataset is not None:
                dataset.map(fn)

    def __getitem__(self, key):
        assert key in self.modes

        return getattr(self, key)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(train={self.train}"
            f", val={self.val}, test={self.test}"
            f", dirname={self.dirname}, filename={self.filename})"
        )

    @classmethod
    def from_dir(cls: Type[T], dirname: str) -> T:
        raise NotImplementedError()


class JSONDatasets(Datasets):
    T = TypeVar("T", bound="JSONDatasets")

    files = {
        "train": "train.json",
        "val": "val.json",
        "test": "test.json",
    }

    @classmethod
    def format(cls, x: Any) -> Any:
        return x

    @classmethod
    def from_dir(cls: Type[T], dirname: str, array: bool = False) -> T:
        # pylint: disable=arguments-differ

        data = {
            key: Dataset.from_json_iter(
                os.path.join(dirname, value), cls.format, array=array
            )
            for key, value in cls.files.items()
        }

        return cls(train=data["train"], val=data["val"], test=data["test"])
