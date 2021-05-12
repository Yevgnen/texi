# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from typing import Optional, Union

import torch.nn as nn
import torch.optim as optim
from transformers.optimization import AdamW, get_linear_schedule_with_warmup


def plm_path(path: Union[str, os.PathLike]) -> str:
    root = os.getenv("PRETRAINED_MODEL_DIR")
    if root:
        return os.path.join(root, path)

    return str(path)


def get_pretrained_optimizer_and_scheduler(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    warmup_steps: int,
    num_training_steps: int,
    optimizer_kwargs: Optional[dict] = None,
    scheduler_kwargs: Optional[dict] = None,
) -> tuple[optim.Optimizer, optim.lr_scheduler.LambdaLR]:
    if lr <= 0:
        raise ValueError("`lr` must be positive.")

    if warmup_steps <= 0:
        raise ValueError("`warmup_steps` must be positive")

    if not optimizer_kwargs:
        optimizer_kwargs = {}

    if not scheduler_kwargs:
        scheduler_kwargs = {}

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, **optimizer_kwargs)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
        **scheduler_kwargs
    )

    return (optimizer, lr_scheduler)
