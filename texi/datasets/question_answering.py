# -*- coding: utf-8 -*-

from typing import Dict

from texi.datasets.dataset import JSONDatasets


class Baike2018QA(JSONDatasets):
    files = {
        "train": "baike_qa_train.json",
        "val": "baike_qa_valid.json",
    }


class ZhidaoQA(JSONDatasets):
    files = {
        "train": "zhidao_qa.json",
    }

    @classmethod
    def format(cls, x: Dict) -> Dict:
        x["id"] = x.pop("_id").pop("$oid")

        return x
