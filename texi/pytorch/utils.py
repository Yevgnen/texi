# -*- coding: utf-8 -*-

from __future__ import annotations

import inspect
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Optional, Union, cast

import ignite.distributed as idist
import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.handlers.checkpoint import Checkpoint
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler
from torchnlp.samplers import BucketBatchSampler

from texi.datasets.dataset import Dataset, Datasets

if TYPE_CHECKING:
    from texi.pytorch.dataset import Collator


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
    sizes = [t.size() for t in tensors]
    if not sizes:
        raise ValueError("At least one tensor must be given.")

    for prev, this in zip(sizes, sizes[1:]):
        if prev[dim] != this[dim]:
            raise ValueError(
                f"Size of dimension `dim` = {dim}"  # type: ignore
                " must be same for all input tensors, "
                f"got: {list(map(tuple, sizes))}."
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


def check_tensors_dimension(tensors: Iterable[torch.Tensor], dim: int):
    if any(not isinstance(x, torch.Tensor) or x.ndim != dim for x in tensors):
        raise ValueError(f"`tensors` must be {dim}d tensors.")


def pad_stack_1d(tensors: Sequence[torch.Tensor], length: int) -> torch.Tensor:
    # NOTE: Works for 1d tensors with size 0.

    check_tensors_dimension(tensors, 1)

    return torch.stack([F.pad(x, [0, length - len(x)]) for x in tensors])


def pad_stack_2d(
    tensors: Sequence[torch.Tensor], max_rows: int, max_columns: int
) -> torch.Tensor:
    # NOTE: Works for 2d tensors with size 0.

    check_tensors_dimension(tensors, 2)

    # https://discuss.pytorch.org/t/padding-zero-size-tensors/118777
    if max_rows == 0:
        return tensors[0].new_zeros(len(tensors), max_rows, max_columns)

    return torch.stack(
        [
            torch.nn.functional.pad(
                x, [0, max_columns - x.size(1), 0, max_rows - x.size(0)]
            )
            for x in tensors
        ]
    )


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


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    collate_fn: Optional[Union[Callable, Collator]] = None,
    drop_last: bool = False,
    sort_key: Callable = lambda x: x,
    **kwargs,
) -> DataLoader:
    sampler = get_sampler(
        cast(Sequence, dataset.examples),
        dataset.is_train(),
        batch_size,
        drop_last=drop_last,
        sort_key=lambda index: sort_key(dataset[index]),
    )

    dataloader = idist.auto_dataloader(
        dataset, batch_sampler=sampler, collate_fn=collate_fn, **kwargs
    )  # type: DataLoader[Dataset]

    return dataloader


def get_dataloaders(
    datasets: Union[Datasets, Mapping[str, Dataset]],
    train_batch_size: Optional[int] = None,
    eval_batch_size: Optional[int] = None,
    batch_size: Optional[int] = None,
    drop_last: bool = False,
    sort_key: Callable = lambda x: x,
    **kwargs,
) -> dict[str, DataLoader]:
    # 1. Train dataset has individual batch size.
    # 2. `drop_last` will alwarys be False for val and test datasets.
    # 3. `sort_key` is passed only in train dataset.

    if (train_batch_size is None or eval_batch_size is None) and batch_size is None:
        raise ValueError(
            "`batch_size` must not be None"
            " if `train_batch_size` or `eval_batch_size` is None"
        )

    if train_batch_size is None:
        train_batch_size = cast(int, batch_size)

    if eval_batch_size is None:
        eval_batch_size = cast(int, batch_size)

    batch_sizes = {
        "train": train_batch_size,
        "val": eval_batch_size,
        "test": eval_batch_size,
    }

    loaders = {}
    for mode, dataset in datasets.items():
        if mode == "train":
            loader = get_dataloader(
                dataset,
                batch_sizes[mode],
                drop_last=drop_last,
                sort_key=sort_key,
                **kwargs,
            )
        else:
            loader = get_dataloader(
                dataset, batch_sizes[mode], drop_last=False, **kwargs
            )

        loaders[mode] = loader

    return loaders


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
