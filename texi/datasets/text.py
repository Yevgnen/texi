# -*- coding: utf-8 -*-

from texi.datasets.dataset import JSONDatasets


class News2016Zh(JSONDatasets):
    files = {
        "train": "news2016zh_train.json",
        "val": "news2016zh_valid.json",
    }
