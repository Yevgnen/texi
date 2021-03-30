# -*- coding: utf-8 -*-

from texi.datasets import classification, text, translation
from texi.datasets.classification import (
    AFQMC,
    CCKS2018,
    CHIP2019,
    LCQMC,
    NCOV2019,
    PAWSX,
    BQCorpus,
    ChineseSNLI,
    ChineseSTSB,
    Sohu2021,
    THUCNews,
)
from texi.datasets.dataset import Dataset, Datasets, JSONDatasets
from texi.datasets.text import News2016Zh
from texi.datasets.translation import Translate2019Zh

__all__ = [
    "Dataset",
    "Datasets",
    "JSONDatasets",
    "classification",
    "CHIP2019",
    "NCOV2019",
    "LCQMC",
    "BQCorpus",
    "PAWSX",
    "AFQMC",
    "CCKS2018",
    "ChineseSNLI",
    "ChineseSTSB",
    "THUCNews",
    "Sohu2021",
    "translation",
    "Translate2019Zh",
    "text",
    "News2016Zh",
]
