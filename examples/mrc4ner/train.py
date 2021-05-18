# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import functools
from typing import Union

import ignite.distributed as idist
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from transformers import AdamW, BertTokenizer, BertTokenizerFast

from texi.apps.ner import split_example
from texi.datasets import JSONDatasets
from texi.datasets.dataset import Dataset, Datasets
from texi.pytorch.plm.mrc4ner import Mrc4Ner, Mrc4NerDataset, Mrc4NerLoss, Mrc4NerParams
from texi.pytorch.plm.mrc4ner.training import eval_step, get_metrics, train_step
from texi.pytorch.plm.utils import get_pretrained_optimizer_and_scheduler, plm_path
from texi.pytorch.training.training import (
    create_engines,
    describe_dataflows,
    run,
    setup_env,
)
from texi.utils import ModeKeys


def get_dataset(
    examples: Dataset,
    tokenizer: BertTokenizerFast,
    params: Mrc4NerParams,
    mode: ModeKeys,
) -> Mrc4NerDataset:
    dataset = Mrc4NerDataset(
        examples,
        params.max_entity_length,
        params.queries,
        tokenizer,
        mode=mode,
        device=idist.device(),
    )

    return dataset


def get_dataflows(
    datasets: Datasets,
    tokenizer: Union[BertTokenizerFast, BertTokenizer],
    params: Mrc4NerParams,
) -> dict[str, DataLoader]:
    # `pin_memory = False` is required since `auto_dataloader` set
    # `pin_memory` to True by default, but we have moved tensors to GPU
    # by passing `device` to Dataset.
    dataflows = Mrc4NerDataset.get_dataloaders(
        {
            mode: get_dataset(
                dataset,
                tokenizer,
                params,
                ModeKeys.TRAIN if mode == "train" else ModeKeys.EVAL,
            )
            for mode, dataset in datasets.items()
            if dataset is not None
        },
        train_batch_size=params["train_batch_size"],
        eval_batch_size=params["eval_batch_size"],
        num_workers=params["num_workers"],
        sort_key=lambda x: 1,
        pin_memory=False,
    )

    return dataflows


def initialize(
    params: Mrc4NerParams,
    num_train_examples: int,
) -> tuple[Mrc4Ner, Mrc4NerLoss, AdamW, LambdaLR]:
    model = Mrc4Ner(params["pretrained_model"])

    num_training_steps = (
        num_train_examples // params["train_batch_size"] * params["max_epochs"]
    )
    warmup_steps = params["lr_warmup"] * num_training_steps
    optimizer, lr_scheduler = get_pretrained_optimizer_and_scheduler(
        model, params["lr"], params["weight_decay"], warmup_steps, num_training_steps
    )
    criteria = Mrc4NerLoss()

    model = idist.auto_model(model)
    optimizer = idist.auto_optim(optimizer)
    criteria = criteria.to(idist.device())

    return model, criteria, optimizer, lr_scheduler


def training(local_rank: int, params: Mrc4NerParams) -> None:
    if idist.get_rank() == 0:
        setup_env(params)

    # Load datasets.
    datasets = JSONDatasets.from_dir(params.data_dir, array=True).load()
    if params.split_delimiter:
        datasets.split(
            functools.partial(
                split_example, delimiters=params.split_delimiter, ignore_errors=True
            )
        )

    if params.max_length > 0:
        datasets.mask(lambda x: len(x["tokens"]) < params["max_length"])

    # Get text/label encoders.
    tokenizer = BertTokenizerFast.from_pretrained(plm_path(params["pretrained_model"]))

    # Get data dataflows.
    dataflows = get_dataflows(datasets, tokenizer, params)
    describe_dataflows(dataflows)

    # Create model.
    model, criteria, optimizer, lr_scheduler = initialize(params, len(datasets.train))

    trainer, evaluators, loggers = create_engines(
        params,
        train_step,
        eval_step,
        dataflows,
        model,
        criteria,
        optimizer,
        lr_scheduler,
        train_metrics=get_metrics(train=True),
        eval_metrics=get_metrics(train=False),
        with_handlers=True,
    )

    # Train!
    trainer.run(dataflows["train"], max_epochs=params["max_epochs"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--params", type=Mrc4NerParams.from_yaml, default="mrc4ner.yaml"
    )

    return parser.parse_args()


if __name__ == "__main__":
    run(training, parse_args().params)
