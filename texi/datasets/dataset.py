# -*- coding: utf-8 -*-

from typing import Dict, Iterable, Optional


class Dataset(object):
    def __init__(self, examples: Iterable[Dict]):
        self.examples = list(examples)

    def __getitem__(self, key):
        return self.examples[key]

    def __iter__(self):
        yield from iter(self.examples)

    def __len__(self):
        return len(self.examples)


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
