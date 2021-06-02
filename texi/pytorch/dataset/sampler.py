# -*- coding: utf-8 -*-

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable
from typing import Optional, Union, cast

import ignite.distributed as idist
import torch
from ignite.distributed.auto import DistributedProxySampler
from torch.utils.data import BatchSampler, Dataset, IterableDataset
from torch.utils.data.sampler import RandomSampler, Sampler


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


class BucketIterableDataset(IterableDataset):
    def __init__(
        self,
        iterator: Union[Iterable, Callable],
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
        if callable(self.iterator):
            iterator = self.iterator()
        else:
            iterator = self.iterator

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            bucket = list(itertools.islice(iterator, 0, self.bucket_size))
        else:
            bucket = list(
                itertools.islice(
                    iterator,
                    worker_info.id,
                    self.bucket_size,
                    worker_info.num_workers,
                )
            )

        if bucket:
            bucket.sort(key=self.sort_key)

            for batch in _random_access_bucket(bucket, self.batch_size, False):
                yield from batch

    def __getitem__(self, index):
        raise NotImplementedError()


def bucket_iterator_dataset(
    iterator: Iterable,
    batch_size: int,
    sort_key: Callable = _identity,
    batch_size_multiplier: int = 100,
) -> BucketIterableDataset:
    return BucketIterableDataset(
        iterator=lambda: itertools.islice(
            iterator, idist.get_rank(), None, idist.get_world_size()
        ),
        batch_size=batch_size,
        sort_key=sort_key,
        batch_size_multiplier=batch_size_multiplier,
    )
