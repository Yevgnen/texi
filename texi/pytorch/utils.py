# -*- coding: utf-8 -*-

import inspect
from typing import Callable, Dict, Sequence, Union

import torch
from torch.utils.data import BatchSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torchnlp.samplers import BucketBatchSampler


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
    train: bool,
    batch_size: int,
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
