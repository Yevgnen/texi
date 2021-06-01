# -*- coding: utf-8 -*-

from __future__ import annotations

import itertools
import random
from collections.abc import Callable, Iterable
from typing import Optional

import torch
from torch.utils.data import BatchSampler, IterableDataset
from torch.utils.data.sampler import Sampler


def _identity(x):
    return x


def random_access_bucket(
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

            yield from random_access_bucket(bucket, self.batch_size, self.drop_last)


class BucketIterableDataset(IterableDataset):
    def __init__(
        self,
        iterator: Iterable,
        batch_size: int,
        sort_key: Callable = _identity,
        batch_size_multiplier: int = 100,
        local_rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.iterator = iterator
        self.batch_size = batch_size
        self.bucket_size = batch_size_multiplier * batch_size
        self.sort_key = sort_key
        if ((local_rank is None) + (world_size is None)) % 2 != 0:
            raise ValueError(
                "`local_rank` and `world_size` must both given or unspecified"
            )
        if local_rank is not None:
            self.iterator = itertools.islice(
                self.iterator, local_rank, None, world_size
            )

        self.local_rank = local_rank
        self.world_size = world_size

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

            yield from random_access_bucket(bucket, self.batch_size, False)

    def __getitem__(self, index):
        raise NotImplementedError()
