# -*- coding: utf-8 -*-

from texi.datasets import classification
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
    THUCNews,
)
from texi.datasets.dataset import Dataset, Datasets

__all__ = [
    "Dataset",
    "Datasets",
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
]
