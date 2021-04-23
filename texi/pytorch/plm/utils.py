# -*- coding: utf-8 -*-

from typing import Dict, Tuple

import torch.nn as nn
import torch.optim as optim
from transformers.optimization import AdamW, get_linear_schedule_with_warmup


def get_pretrained_optimizer_and_scheduler(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    warmup_steps: int,
    num_training_steps: int,
    optimizer_kwargs: Dict,
    scheduler_kwargs: Dict,
) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
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
