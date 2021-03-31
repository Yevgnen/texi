# -*- coding: utf-8 -*-

import argparse
import logging
import re
from pathlib import Path

from texi.datasets import (
    Baike2018QA,
    News2016Zh,
    ToutiaoNews,
    Translate2019Zh,
    WebText2019Zh,
    WeixinPublicCorpus,
    ZhidaoQA,
)

logger = logging.getLogger(__name__)


def write_texts(text_iter, filename, preprocess=lambda x: x):
    wrote = 0
    with open(filename, mode="w") as f:
        for text in text_iter:
            f.writelines(preprocess(text) + "\n")
            wrote += 1

    return wrote


def iter_news2016zh(dirname):
    datasets = News2016Zh.from_dir(dirname)
    for _, dataset in datasets.items():
        if dataset is not None:
            for x in dataset:
                yield x["title"] + x["content"]


def iter_baike2018qa(dirname):
    datasets = Baike2018QA.from_dir(dirname)
    for _, dataset in datasets.items():
        if dataset is not None:
            for x in dataset:
                yield x["title"] + x["desc"] + x["answer"]


def iter_toutiaonews(dirname):
    datasets = ToutiaoNews.from_dir(dirname)
    for _, dataset in datasets.items():
        if dataset is not None:
            for x in dataset:
                yield x["title"]


def iter_translation2019zh(dirname):
    datasets = Translate2019Zh.from_dir(dirname)
    for _, dataset in datasets.items():
        if dataset is not None:
            for x in dataset:
                yield x["chinese"]


def iter_webtext2019zh(dirname):
    datasets = WebText2019Zh.from_dir(dirname)
    for _, dataset in datasets.items():
        if dataset is not None:
            for x in dataset:
                yield x["title"] + x["desc"] + x["content"]


def iter_weixin_public_corpus(dirname):
    datasets = WeixinPublicCorpus.from_dir(dirname)
    for _, dataset in datasets.items():
        if dataset is not None:
            for x in dataset:
                yield x["title"] + x["content"]


def iter_zhidaoqa(dirname):
    datasets = ZhidaoQA.from_dir(dirname)
    for _, dataset in datasets.items():
        if dataset is not None:
            for x in dataset:
                yield x["question"] + x["answer"]


def preprocess(x):
    x = re.sub(r"[\r\n]", "", x)

    return x


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default="./output")

    return parser.parse_args()  # pylint: disable=redefined-outer-name


def main(args):
    data = {
        "new2016zh": iter_news2016zh,
        "baike2018qa": iter_baike2018qa,
        "toutiao-text-classfication-dataset": iter_toutiaonews,
        "translation2019zh": iter_translation2019zh,
        "webtext2019zh": iter_webtext2019zh,
        "weixin_public_corpus": iter_weixin_public_corpus,
        "zhidao_qa": iter_zhidaoqa,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dataset, iterator in data.items():
        lines = write_texts(
            iterator(args.data_dir / dataset),
            args.output_dir / dataset,
            preprocess=preprocess,
        )
        logger.info("Processed %d lines of: %s", lines, dataset)


if __name__ == "__main__":
    main(parse_args())
