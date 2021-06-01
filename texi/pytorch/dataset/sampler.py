# -*- coding: utf-8 -*-

from __future__ import annotations

import itertools
import random
from collections.abc import Callable, Iterable
from typing import Optional

import ignite.distributed as idist
import torch
from ignite.distributed.auto import DistributedProxySampler
from torch.utils.data import BatchSampler, IterableDataset
from torch.utils.data.sampler import Sampler


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
            start = random.randint(0, size - batch_size)
            end = start + batch_size
            yield bucket[start:end]
            bucket = bucket[:start] + bucket[end:]


class BucketBatchSampler(BatchSampler):
    def __init__(
        self,
        sampler: Sampler,
        batch_size: int,
        drop_last: bool,
        sort_key: Callable = _identity,
        batch_size_multiplier: int = 100,
    ) -> None:
        super().__init__(sampler, batch_size, drop_last)

        self.bucket_size = min(
            batch_size_multiplier * batch_size, len(sampler)  # type: ignore
        )
        self.batch_sampler = BatchSampler(sampler, self.bucket_size, False)
        self.sort_key = sort_key

    def __iter__(self):
        for bucket in self.batch_sampler:
            bucket.sort(key=self.sort_key)

            yield from _random_access_bucket(bucket, self.batch_size, self.drop_last)


def bucket_batch_sampler(
    sampler: Sampler,
    batch_size: int,
    drop_last: bool,
    sort_key: Callable = _identity,
    batch_size_multiplier: int = 100,
) -> DistributedProxySampler:
    return DistributedProxySampler(
        BucketBatchSampler(
            sampler,
            batch_size,
            drop_last,
            sort_key=sort_key,
            batch_size_multiplier=batch_size_multiplier,
        ),
        num_replicas=idist.get_world_size(),
        rank=idist.get_rank(),
    )


class BucketIterableDataset(IterableDataset):
    def __init__(
        self,
        iterator: Iterable,
        batch_size: int,
        sort_key: Callable = _identity,
        batch_size_multiplier: int = 100,
    ) -> None:
        super().__init__()
        self.iterator = iterator
        self.batch_size = batch_size
        self.bucket_size = batch_size_multiplier * batch_size
        self.sort_key = sort_key

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            bucket = list(itertools.islice(self.iterator, 0, self.bucket_size))
        else:
            bucket = list(
                itertools.islice(
                    self.iterator,
                    worker_info.id,
                    self.bucket_size,
                    worker_info.num_workers,
                )
            )

        if bucket:
            bucket.sort(key=self.sort_key)

            yield from _random_access_bucket(bucket, self.batch_size, False)

    def __getitem__(self, index):
        raise NotImplementedError()


def _get_local_slice(iterator, world_size=None, rank=None):
    if ((rank is None) + (world_size is None)) % 2 != 0:
        raise ValueError("`rank` and `world_size` must both given or unspecified")
    if world_size is not None:
        iterator = itertools.islice(iterator, rank, None, world_size)

    return iterator


def bucket_iterator_dataset(
    iterator: Iterable,
    batch_size: int,
    sort_key: Callable = _identity,
    batch_size_multiplier: int = 100,
) -> BucketIterableDataset:
    iterator = _get_local_slice(
        iterator, world_size=idist.get_world_size(), rank=idist.get_rank()
    )

    return BucketIterableDataset(
        iterator=iterator,
        batch_size=batch_size,
        sort_key=sort_key,
        batch_size_multiplier=batch_size_multiplier,
    )
