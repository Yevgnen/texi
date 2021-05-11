# -*- coding: utf-8 -*-

from __future__ import annotations

import inspect
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Union

import torch
import torch.nn as nn
from ignite.handlers.checkpoint import Checkpoint
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
) -> list[tuple[torch.Tensor, ...]]:
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
) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
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
    sort_key: Callable = lambda x: x,
) -> Union[BatchSampler, BucketBatchSampler]:
    if train:
        sampler = RandomSampler(examples)  # type: ignore
        batch_sampler = BucketBatchSampler(
            sampler, batch_size=batch_size, drop_last=drop_last, sort_key=sort_key
        )
    else:
        sampler = SequentialSampler(examples)  # type: ignore
        batch_sampler = BatchSampler(
            sampler, batch_size=batch_size, drop_last=drop_last
        )

    return batch_sampler


def get_default_arguments(f: Callable) -> dict:
    return {
        key: value.default
        for key, value in inspect.signature(f).parameters.items()
        if value.default is not value.empty
    }


def load_checkpoint(
    to_load: Union[Mapping, nn.Module], checkpoint: Union[str, os.PathLike]
) -> None:
    if isinstance(to_load, nn.Module):
        to_load = {"model": to_load}

    ckpt = torch.load(checkpoint, map_location="cpu")
    Checkpoint.load_objects(to_load=to_load, checkpoint=ckpt)
