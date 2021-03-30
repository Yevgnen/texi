# -*- coding: utf-8 -*-

import glob
import json
import os

from texi.datasets.dataset import Datasets, JSONDatasets


class News2016Zh(JSONDatasets):
    files = {
        "train": "news2016zh_train.json",
        "val": "news2016zh_valid.json",
    }


class WebText2019Zh(JSONDatasets):
    files = {
        "train": "web_text_zh_train.json",
        "val": "web_text_zh_valid.json",
        "test": "web_text_zh_testa.json",
    }


class Wiki2019Zh(Datasets):
    @classmethod
    def from_dir(cls, dirname: str):
        examples = []
        for filename in glob.iglob(os.path.join(dirname, "**/*"), recursive=True):
            if os.path.isdir(filename):
                continue

            with open(filename) as f:
                for line in f:
                    line = line.rstrip()
                    if line:
                        example = json.loads(line)
                        examples += [example]

        return cls(train=examples)


class WeixinPublicCorpus(JSONDatasets):
    files = {
        "train": "articles.json",
    }
