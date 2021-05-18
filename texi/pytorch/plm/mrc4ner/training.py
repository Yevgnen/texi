# -*- coding: utf-8 -*-

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import ignite.distributed as idist
import torch.nn as nn
from ignite.engine import Engine

from texi.pytorch.metrics import NerMetrics, ReMetrics
from texi.pytorch.training.params import Params
from texi.pytorch.training.training import Metrics

if TYPE_CHECKING:
    from texi.pytorch.plm.mrc4ner.model import Mrc4Ner


class Mrc4NerParams(Params):
    # pylint: disable=too-many-instance-attributes
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pretrained_model = kwargs.get("pretrained_model", "bert-base-uncased")
        self.max_entity_length = kwargs.get("max_entity_length", 10)
        self.queries = kwargs["queries"]
        self.span_filter_threshold = kwargs.get("span_filter_threshold", 0.4)
        self.token_delimiter = kwargs.get("token_delimiter", " ")
        self.split_delimiter = kwargs.get("split_delimiter")
        self.max_length = kwargs.get("max_length", -1)


def train_step(_: Engine, model: Mrc4Ner, batch: Mapping, criteria: nn.Module) -> dict:
    output = model(
        batch["input_ids"],
        batch["attention_mask"],
        batch["token_type_ids"],
        batch["span_index"],
    )

    loss = criteria(
        batch["start"],
        batch["end"],
        batch["span_label"],
        batch["span_mask"],
        output["start_logit"],
        output["end_logit"],
        output["span_logit"],
    )

    return {"batch": batch, "loss": loss}


def eval_step(_: Engine, model: Mrc4Ner, batch: Mapping) -> dict:
    _.state.metrics = {
        'NER.micro.f1': 1,
    }
    return {
        'NER.micro.f1': 1,
    }


def get_metrics(train: bool = True) -> Metrics:
    if train:
        return {}

    return {}
