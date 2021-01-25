# -*- coding: utf-8 -*-

import inspect
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import BatchSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torchnlp.samplers import BucketBatchSampler
from transformers.optimization import AdamW, get_linear_schedule_with_warmup


def cuda(enable: bool) -> bool:
    return enable and torch.cuda.is_available()


def device(
    device_str: str = "", return_string: bool = True
) -> Union[str, torch.device]:
    if (
        not device_str
        or device_str.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        device_str = "cpu"

    return device_str if return_string else torch.device(device_str)


def get_sampler(
    examples: Sequence,
    train: bool = True,
    batch_size: int = 32,
    drop_last: bool = False,
) -> BatchSampler:
    if train:
        sampler_class = RandomSampler
        batch_sampler_class = BucketBatchSampler
    else:
        sampler_class = SequentialSampler
        batch_sampler_class = BatchSampler

    sampler = sampler_class(examples)
    batch_sampler = batch_sampler_class(
        sampler, batch_size=batch_size, drop_last=drop_last
    )

    return batch_sampler


def get_default_arguments(f: Callable) -> Dict:
    return {
        key: value.default
        for key, value in inspect.signature(f).parameters.items()
        if value.default is not value.empty
    }


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


def length_to_mask(
    lengths: torch.Tensor,
    max_len: Optional[Union[int, torch.Tensor]] = None,
    batch_first: bool = False,
) -> torch.Tensor:
    if max_len is None:
        max_len = lengths.max()

    mask = torch.arange(max_len, device=lengths.device).unsqueeze(dim=1).expand(
        max_len, len(lengths)
    ) < lengths.unsqueeze(dim=0)

    return mask.transpose(0, 1) if batch_first else mask


def mask_to_length(masks: torch.Tensor, batch_first: bool = False) -> torch.Tensor:
    return masks.sum(dim=int(batch_first)).long()
