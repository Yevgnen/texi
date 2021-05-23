# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping
from typing import Union

import ignite.distributed as idist
import torch
import torch.nn as nn
from ignite.engine import Engine
from ignite.metrics import Accuracy, Fbeta, Precision, Recall
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AdamW, BertTokenizer, BertTokenizerFast

from texi.datasets import JSONDatasets
from texi.datasets.dataset import Dataset, Datasets
from texi.pytorch.models import BertForSequenceClassification
from texi.pytorch.plm.dataset.classification import TextClassificationCollator
from texi.pytorch.plm.utils import get_pretrained_optimizer_and_scheduler, plm_path
from texi.pytorch.training.params import Params as _Params
from texi.pytorch.training.training import (
    create_engines,
    describe_dataflows,
    run,
    setup_env,
)
from texi.pytorch.utils import get_dataloader


class Params(_Params):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pretrained_model = kwargs.get("pretrained_model", "hfl/chinese-bert-wwm")
        self.dropout = kwargs.get("dropout", 0.1)
        self.pooling = kwargs.get("pooling", "mean")
        self.num_labels = kwargs.get("num_labels", 2)


def get_dataflows(
    datasets: Datasets,
    tokenizer: Union[BertTokenizerFast, BertTokenizer],
    params: Params,
) -> dict[str, DataLoader]:
    # `pin_memory = False` is required since `auto_dataloader` set
    # `pin_memory` to True by default, but we have moved tensors to GPU
    # by passing `device` to Dataset.
    dataflows = {
        mode: get_dataloader(
            dataset,
            params[f"{Dataset.map_modekeys(mode)}_batch_size"],
            collate_fn=TextClassificationCollator(dataset, tokenizer, idist.device()),
            num_workers=params["num_workers"],
            sort_key=lambda x: len(x["text"]),
            pin_memory=False,
        )
        for mode, dataset in datasets.items()
    }

    return dataflows


def initialize(
    params: Params,
    num_train_examples: int,
) -> tuple[nn.Module, nn.BCEWithLogitsLoss, AdamW, LambdaLR]:
    model = BertForSequenceClassification(
        params["pretrained_model"],
        dropout=params["dropout"],
        pooling=params["pooling"],
        num_labels=params["num_labels"],
    )

    num_training_steps = (
        num_train_examples // params["train_batch_size"] * params["max_epochs"]
    )
    warmup_steps = params["lr_warmup"] * num_training_steps
    optimizer, lr_scheduler = get_pretrained_optimizer_and_scheduler(
        model, params["lr"], params["weight_decay"], warmup_steps, num_training_steps
    )

    criteria = nn.BCEWithLogitsLoss()
    model = idist.auto_model(model)
    optimizer = idist.auto_optim(optimizer)
    criteria = criteria.to(idist.device())

    return model, criteria, optimizer, lr_scheduler


def train_step(
    _: Engine, model: nn.Module, batch: Mapping, criteria: nn.Module
) -> dict:
    x, y = batch

    output = model(
        x["input_ids"],
        x["attention_mask"],
        x["token_type_ids"],
    )

    loss = criteria(output.squeeze(dim=-1), y.float())

    return {"loss": loss}


def eval_step(
    _: Engine, model: nn.Module, batch: Mapping
) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = batch

    output = model(
        x["input_ids"],
        x["attention_mask"],
        x["token_type_ids"],
    )

    return torch.sigmoid(output.squeeze(dim=-1)).round(), y.float()


def training(local_rank: int, params: Params) -> None:
    if idist.get_rank() == 0:
        setup_env(params)

    # Load datasets.
    datasets = JSONDatasets.from_dir(params.data_dir, array=True).load()
    tokenizer = BertTokenizerFast.from_pretrained(plm_path(params["pretrained_model"]))

    # Get data dataflows.
    dataflows = get_dataflows(datasets, tokenizer, params)
    describe_dataflows(dataflows)

    # Create model.
    model, criteria, optimizer, lr_scheduler = initialize(params, len(datasets.train))

    accuracy = Accuracy(device=idist.device())
    precision = Precision(average=False, device=idist.device())
    recall = Recall(average=False, device=idist.device())
    f1 = Fbeta(1.0, precision=precision, recall=recall, device=idist.device())

    # Create engines
    trainer, *_ = create_engines(
        params,
        train_step,
        eval_step,
        dataflows,
        model,
        criteria,
        optimizer,
        lr_scheduler,
        eval_metrics={
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        with_handlers=True,
    )

    # Train!
    trainer.run(dataflows["train"], max_epochs=params["max_epochs"])


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--params", type=Params.from_yaml, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    run(training, parse_args().params)
