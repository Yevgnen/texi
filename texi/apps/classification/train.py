# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from collections.abc import Mapping
from typing import Union

import ignite.distributed as idist
import torch
import torch.nn as nn
from ignite.engine import Engine
from ignite.metrics import Accuracy, Fbeta, Precision, Recall
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertTokenizerFast

from texi.datasets import JSONDatasets
from texi.datasets.dataset import Dataset, Datasets
from texi.preprocessing import LabelEncoder
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
        self.model_name = kwargs["model_name"]
        self.num_labels = kwargs["num_labels"]
        if self.num_labels < 2:
            raise ValueError("`num_labels` must >= 2")
        self.pretrained_model = kwargs.get("pretrained_model", "hfl/chinese-bert-wwm")
        self.dropout = kwargs.get("dropout", 0.1)
        self.pooling = kwargs.get("pooling", "mean")


def get_dataflows(
    datasets: Datasets,
    tokenizer: Union[BertTokenizerFast, BertTokenizer],
    label_encoder: LabelEncoder,
    params: Params,
) -> dict[str, DataLoader]:
    # `pin_memory = False` is required since `auto_dataloader` set
    # `pin_memory` to True by default, but we have moved tensors to GPU
    # by passing `device` to Dataset.
    dataflows = {
        mode: get_dataloader(
            dataset,
            params[f"{Dataset.map_modekeys(mode)}_batch_size"],
            collate_fn=TextClassificationCollator(
                dataset, tokenizer, label_encoder, idist.device()
            ),
            num_workers=params["num_workers"],
            sort_key=lambda x: len(x["text"]),
            pin_memory=False,
        )
        for mode, dataset in datasets.items()
    }

    return dataflows


def get_model(params: Params) -> nn.Module:
    name = params["model_name"].lower()
    if name == "bert":
        model = BertForSequenceClassification(
            params["pretrained_model"],
            dropout=params["dropout"],
            pooling=params["pooling"],
            num_labels=params["num_labels"],
        )
    else:
        raise KeyError(name)

    return model


def get_criteria(params: Params) -> nn.Module:
    if params["num_labels"] == 2:
        return nn.BCEWithLogitsLoss()

    return nn.CrossEntropyLoss()


def initialize(
    params: Params, num_train_examples: int
) -> tuple[nn.Module, nn.Module, Optimizer, _LRScheduler]:
    model = get_model(params)

    criteria = get_criteria(params)

    num_training_steps = (
        num_train_examples // params["train_batch_size"] * params["max_epochs"]
    )
    warmup_steps = params["lr_warmup"] * num_training_steps
    optimizer, lr_scheduler = get_pretrained_optimizer_and_scheduler(
        model, params["lr"], params["weight_decay"], warmup_steps, num_training_steps
    )

    model = idist.auto_model(model)
    criteria = criteria.to(idist.device())
    optimizer = idist.auto_optim(optimizer)

    return model, criteria, optimizer, lr_scheduler


def train_step(
    _: Engine, model: nn.Module, batch: Mapping, criteria: nn.Module
) -> dict:
    x, y = batch

    logit = model(**x)

    loss = criteria(logit, y.float())

    return {"loss": loss}


def eval_step(
    _: Engine, model: nn.Module, batch: Mapping
) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = batch

    logit = model(**x)

    if logit.ndim == 1:
        y_pred = torch.sigmoid(logit.squeeze(dim=-1)).round()
    else:
        y_pred = torch.argmax(logit, dim=-1)

    return y_pred, y.float()


def training(local_rank: int, params: Params) -> None:
    if idist.get_rank() == 0:
        setup_env(params)

    # Load datasets.
    datasets = JSONDatasets.from_dir(params.data_dir, array=True).load()
    tokenizer = BertTokenizerFast.from_pretrained(plm_path(params["pretrained_model"]))
    label_encoder = LabelEncoder(x["label"] for x in datasets.train)

    # Get data dataflows.
    dataflows = get_dataflows(datasets, tokenizer, label_encoder, params)
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