# -*- coding: utf-8 -*-

from texi.datasets.dataset import JSONDatasets


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
