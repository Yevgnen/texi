# -*- coding: utf-8 -*-

from texi.datasets.dataset import JSONDatasets


class Translate2019Zh(JSONDatasets):
    files = {
        "train": "translation2019zh_train.json",
        "val": "translation2019zh_train.json",
    }
