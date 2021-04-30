# -*- coding: utf-8 -*-

import inspect
from typing import Callable, Dict, Iterable, List, Sequence, Tuple, Union

import torch
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler
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


def split_tensors(
    tensors: Iterable[torch.Tensor], chunk_size: int, dim: int = 0
) -> List[Tuple[torch.Tensor, ...]]:
    if len(set(x.size(dim=dim) for x in tensors)) != 1:
        raise ValueError(
            f"Size of dimension `dim` = {dim} must be same for all input tensors"
        )

    splits = list(zip(*(torch.split(x, chunk_size, dim=dim) for x in tensors)))

    return splits


def split_apply(
    f: Callable,
    inputs: Iterable[torch.Tensor],
    chunk_size: int,
    *args,
    dim: int = 0,
    **kwargs,
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    chunks = split_tensors(inputs, chunk_size=chunk_size, dim=dim)
    outputs = [f(*chunk, *args, **kwargs) for chunk in chunks]

    if all(isinstance(t, torch.Tensor) for t in outputs):
        return torch.cat(outputs, dim=dim)

    outputs = list(zip(*outputs))
    tuple_outputs = tuple(torch.cat(x, dim=dim) for x in outputs)

    return tuple_outputs


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
