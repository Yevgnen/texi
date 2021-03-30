# -*- coding: utf-8 -*-

from texi.datasets.dataset import JSONDatasets


class Baike2018QA(JSONDatasets):
    files = {
        "train": "baike_qa_train.json",
        "val": "baike_qa_valid.json",
    }
