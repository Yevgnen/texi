# -*- coding: utf-8 -*-

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable
from typing import Optional, cast

import ignite.distributed as idist
import torch
from ignite.distributed.auto import DistributedProxySampler
from torch.utils.data import BatchSampler, Dataset
from torch.utils.data import IterableDataset as _IterableDataset
from torch.utils.data.sampler import RandomSampler, Sampler

from texi.utils import ModeKeys


def _identity(x):
    return x


def _random_access_bucket(
    bucket: list, batch_size: int, drop_last: Optional[bool] = None
) -> Iterable:
    while bucket:
        size = len(bucket)
        if size <= batch_size:
            if size == batch_size or drop_last is False:
                yield bucket
            bucket = []
        else:
            # NOTE: Use `torch.randint` becasue
            # `DistributedProxySampler` use `torch.manual_seed`.
            start = cast(int, torch.randint(0, size - batch_size, size=(1,)).item())
            end = start + batch_size
            yield bucket[start:end]
            bucket = bucket[:start] + bucket[end:]


class BucketBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset: Dataset,
        sampler: Sampler,
        batch_size: int,
        drop_last: bool,
        sort_key: Callable = _identity,
        batch_size_multiplier: int = 100,
    ) -> None:
        super().__init__(sampler, batch_size, drop_last)

        self.dataset = dataset
        self.bucket_size = min(
            batch_size_multiplier * batch_size, len(sampler)  # type: ignore
        )
        self.batch_sampler = BatchSampler(sampler, self.bucket_size, False)
        self.sort_key = sort_key

    def __iter__(self):
        for bucket in self.batch_sampler:
            bucket.sort(key=lambda i: self.sort_key(self.dataset[i]))

            yield from _random_access_bucket(bucket, self.batch_size, self.drop_last)


def bucket_batch_sampler(
    dataset: Dataset,
    batch_size: int,
    drop_last: bool,
    sort_key: Callable = _identity,
    batch_size_multiplier: int = 100,
) -> DistributedProxySampler:
    return DistributedProxySampler(
        BucketBatchSampler(
            dataset,
            RandomSampler(range(len(dataset))),  # type: ignore
            batch_size,
            drop_last,
            sort_key=sort_key,
            batch_size_multiplier=batch_size_multiplier,
        ),
        num_replicas=idist.get_world_size(),
        rank=idist.get_rank(),
    )


class IterableDataset(_IterableDataset):
    def __init__(
        self,
        generating_function: Callable,
        batch_size: int,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        mode: ModeKeys = ModeKeys.TRAIN,  # TODO: Make wrapper.
    ) -> None:
        super().__init__()
        self.generating_function = generating_function
        self.batch_size = batch_size
        if ((rank is None) + (world_size is None)) % 2 != 0:
            raise ValueError("`rank` and `world_size` must both given or unspecified")
        self.world_size = world_size
        self.rank = rank
        self.mode = mode

    def _get_iterator(self):
        # Generate new iterator.
        iterator = self.generating_function()

        # Deal with `DistributedDataParallel`.
        if self.world_size is not None and self.rank is not None:
            iterator = itertools.islice(iterator, self.rank, None, self.world_size)

        # Deal with `num_workers` > 1 of `DataLoader`.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            iterator = itertools.islice(
                iterator, worker_info.id, None, worker_info.num_workers
            )

        return iterator

    def __getitem__(self, index):
        raise RuntimeError("`IterableDataset` does not support indexing")

    def __iter__(self):
        yield from self._get_iterator()


class BucketIterableDataset(IterableDataset):
    def __init__(
        self,
        generating_function: Callable,
        batch_size: int,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        sort_key: Callable = _identity,
        batch_size_multiplier: int = 100,
        mode: ModeKeys = ModeKeys.TRAIN,  # TODO: Make wrapper.
    ) -> None:
        super().__init__(
            generating_function, batch_size, world_size=world_size, rank=rank, mode=mode
        )
        self.bucket_size = batch_size_multiplier * batch_size
        self.sort_key = sort_key

    def __iter__(self):
        iterator = self._get_iterator()

        while bucket := list(itertools.islice(iterator, None, self.bucket_size)):
            bucket.sort(key=self.sort_key)

            for batch in _random_access_bucket(bucket, self.batch_size, False):
                yield from batch
