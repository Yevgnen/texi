# -*- coding: utf-8 -*-

import itertools
import json
import os
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

T_Dataset = TypeVar("T_Dataset", bound="Dataset")
T_Datasets = TypeVar("T_Datasets", bound="Datasets")


class Dataset(object):
    def __init__(self, examples: Union[Iterable[Dict], Callable[[], Iterable[Dict]]]):
        if callable(examples):
            self.load_examples = examples
            self.examples = None
        else:
            self.examples = list(examples)
            self.load_examples = None

    def _check_loaded(self):
        if self.examples is None:
            raise RuntimeError("Dataset is not loaded, call `.load()` first")

    def load(self) -> T_Dataset:
        if callable(self.load_examples) and self.examples is None:
            self.examples = list(self.load_examples())

        return self

    def map(self, fn: Callable[[Mapping], Dict]) -> None:
        self._check_loaded()

        examples = [fn(x) for x in self.examples]
        if examples and isinstance(examples[0], list):
            examples = list(itertools.chain.from_iterable(examples))

        self.examples = examples

    def describe(self) -> Dict[str, Any]:
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
        cls,
        filename: str,
        format_function: Optional[Callable[[Dict], Dict]] = lambda x: x,
        array: bool = False,
    ) -> T_Dataset:
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
    def from_json(cls, filename: str) -> T_Dataset:
        return cls(list(cls.from_json_iter(filename)))


class Datasets(object):
    def __init__(
        self,
        train: Optional[Dataset] = None,
        val: Optional[Dataset] = None,
        test: Optional[Dataset] = None,
        dirname: Optional[str] = None,
        filename: Optional[str] = None,
    ):
        self.train = train
        self.val = val
        self.test = test
        self.dirname = dirname
        self.filename = filename

        self.modes = {"train", "val", "test"}

    def load(self) -> T_Datasets:
        for mode in self.modes:
            dataset = getattr(self, mode)
            if dataset is not None:
                dataset.load()

        return self

    def items(self) -> Iterable[Tuple[str, Dataset]]:
        for mode in self.modes:
            yield mode, getattr(self, mode)

    def map(self, fn: Callable[[Mapping], Dict]) -> None:
        self.train.map(fn)
        self.val.map(fn)
        self.test.map(fn)

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
    def from_dir(cls, dirname: str) -> T_Datasets:
        raise NotImplementedError()


class JSONDatasets(Datasets):
    files = {
        "train": "train.json",
        "val": "val.json",
        "test": "test.json",
    }

    @classmethod
    def format(cls, x: Dict) -> Dict:
        return x

    @classmethod
    def from_dir(cls, dirname: str, array: bool = False) -> T_Datasets:
        data = {
            key: Dataset.from_json_iter(
                os.path.join(dirname, value), cls.format, array=array
            )
            for key, value in cls.files.items()
        }

        return cls(**data)
