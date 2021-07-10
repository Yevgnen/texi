# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from collections.abc import Mapping
from typing import Union

import ignite.distributed as idist
import torch
import torch.nn as nn
from carton.file import iter_lines
from ignite.engine import Engine
from ignite.metrics import Accuracy
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from transformers import BertConfig, BertForMaskedLM, BertTokenizer, BertTokenizerFast

from texi.datasets.dataset import Dataset, Datasets
from texi.pytorch.dataset.plm_collator import MaskedLMCollator
from texi.pytorch.dataset.sampler import BucketIterableDataset, IterableDataset
from texi.pytorch.training.params import Params as _Params
from texi.pytorch.training.training import create_engines, run, setup_env
from texi.pytorch.utils import (
    get_dataloader,
    get_pretrained_optimizer_and_scheduler,
    plm_path,
)
from texi.utils import ModeKeys


class Params(_Params):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pretrained_model = kwargs["pretrained_model"]


def load_datasets(params):
    def _transform(x):
        return {"text": x.split(" ")}

    train = BucketIterableDataset(
        lambda: iter_lines(params["train_data"], transform=_transform),
        params["train_batch_size"],
        world_size=idist.get_world_size(),
        rank=idist.get_rank(),
        sort_key=lambda x: len(x["text"]),
        mode=ModeKeys.TRAIN,
    )
    val = IterableDataset(
        lambda: iter_lines(params["val_data"], transform=_transform),
        params["eval_batch_size"],
        world_size=idist.get_world_size(),
        rank=idist.get_rank(),
        mode=ModeKeys.EVAL,
    )
    test = IterableDataset(
        lambda: iter_lines(params["test_data"], transform=_transform),
        params["predict_batch_size"],
        world_size=idist.get_world_size(),
        rank=idist.get_rank(),
        mode=ModeKeys.PREDICT,
    )

    return {
        "train": train,
        "val": val,
        "test": test,
    }


def get_dataflows(
    datasets: Datasets,
    tokenizer: Union[BertTokenizerFast, BertTokenizer],
    params: Params,
) -> dict[str, DataLoader]:
    dataflows = {
        mode: get_dataloader(
            dataset,
            batch_size=params[f"{Dataset.map_modekeys(mode)}_batch_size"],
            collate_fn=MaskedLMCollator(tokenizer, mode=Dataset.map_modekeys(mode)),
            num_workers=params["num_workers"],
            pin_memory=params["pin_memory"],
        )
        for mode, dataset in datasets.items()
    }

    return dataflows


def initialize(params: Params) -> tuple[nn.Module, nn.Module, Optimizer, _LRScheduler]:
    model = BertForMaskedLM(BertConfig())

    criteria = nn.CrossEntropyLoss()

    num_training_steps = (
        params["train_size"] // params["train_batch_size"] * params["max_epochs"]
    )
    warmup_steps = params["lr_warmup"] * num_training_steps
    warmup_steps = 1
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

    output = model(
        input_ids=x["input_ids"],
        attention_mask=x["attention_mask"],
        token_type_ids=x["token_type_ids"],
    )

    loss = criteria(output["logits"].flatten(0, -2), y.flatten())

    return {"loss": loss}


def eval_step(
    _: Engine, model: nn.Module, batch: Mapping
) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = batch

    output = model(
        input_ids=x["input_ids"],
        attention_mask=x["attention_mask"],
        token_type_ids=x["token_type_ids"],
    )

    mlm_mask = x["mlm_mask"].flatten().bool()
    y_pred = output["logits"].flatten(0, -2)[mlm_mask]
    y = y.flatten()[mlm_mask]

    return y_pred, y


def training(local_rank: int, params: Params) -> None:
    if idist.get_rank() == 0:
        setup_env(params)

    # Load datasets.
    datasets = load_datasets(params)
    tokenizer = BertTokenizerFast.from_pretrained(plm_path(params["pretrained_model"]))

    # Get data dataflows.
    dataflows = get_dataflows(datasets, tokenizer, params)

    # Create model.
    model, criteria, optimizer, lr_scheduler = initialize(params)

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
            "accuracy": Accuracy(device=idist.device()),
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
