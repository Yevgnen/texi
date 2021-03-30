# -*- coding: utf-8 -*-

import json
import os
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

    @classmethod
    def from_json(cls, filename: str):
        examples = []
        with open(filename) as f:
            for line in f:
                line = line.rstrip()
                if line:
                    examples += [json.loads(line)]

        return cls(examples)


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


class JSONDatasets(Datasets):
    files = {}

    @classmethod
    def from_dir(cls, dirname: str):
        data = {
            key: Dataset.from_json(os.path.join(dirname, value))
            for key, value in cls.files.items()
        }

        return cls(**data)
