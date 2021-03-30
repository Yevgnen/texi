# -*- coding: utf-8 -*-

import json
import os
from typing import Callable, Dict, Iterable, Optional, Union


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

    def load(self):
        if callable(self.load_examples) and self.examples is None:
            self.examples = list(self.load_examples())

        return self

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
    def from_json_iter(cls, filename: str):
        def _iter():
            with open(filename) as f:
                for line in f:
                    line = line.rstrip()
                    if line:
                        yield json.loads(line)

        return cls(_iter)

    @classmethod
    def from_json(cls, filename: str):
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

    def load(self):
        for mode in ["train", "val", "test"]:
            dataset = getattr(self, mode)
            if dataset is not None:
                dataset.load()

        return self


class JSONDatasets(Datasets):
    files = {}

    @classmethod
    def from_dir(cls, dirname: str):
        data = {
            key: Dataset.from_json_iter(os.path.join(dirname, value))
            for key, value in cls.files.items()
        }

        return cls(**data)
