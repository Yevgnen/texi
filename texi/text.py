# -*- coding: utf-8 -*-

from __future__ import annotations

import itertools
import math
import multiprocessing
import os
from collections.abc import Callable, Generator, Iterable
from typing import Any, Optional, Union


def _get_chunk_size(filename, num_workers=multiprocessing.cpu_count()):
    return math.ceil(os.path.getsize(filename) / num_workers)


def _get_chunkified_args(
    filename, *args, num_workers=multiprocessing.cpu_count(), chunk_size=None
):
    if not chunk_size:
        chunk_size = _get_chunk_size(filename, num_workers=num_workers)

    return [
        (filename, *args, start, end)
        for filename, *args, (start, end) in zip(
            itertools.repeat(filename),
            *[itertools.repeat(arg) for arg in args],
            chunkify(filename, chunk_size=chunk_size),
        )
    ]


def _iter_line_wrapper(filename, fn, start, end):
    return [fn(line) for line in readlines(filename, start, end)]


def chunkify(
    filename: Union[str, os.PathLike], chunk_size: int = 1024 * 1024
) -> Generator[tuple[int, int], None, None]:
    start = 0
    size = os.path.getsize(filename)
    with open(filename, mode="rb") as f:
        while True:
            f.seek(chunk_size, 1)
            f.readline()
            end = min(f.tell(), size)
            yield start, end

            if end >= size:
                break
            start = end


def readlines(
    filename: Union[str, os.PathLike],
    start: Optional[int] = None,
    end: Optional[int] = None,
    **kwargs
) -> Generator[str, None, None]:
    with open(filename, **kwargs) as f:
        if start:
            f.seek(start)
        while True:
            line = f.readline()
            if end and f.tell() > end or not line:
                break

            yield line


def map_text(
    filename: Union[str, os.PathLike],
    fn: Callable[[str, int, int], Any],
    num_workers: int = multiprocessing.cpu_count(),
    chunk_size: Optional[int] = None,
) -> Iterable:
    args = _get_chunkified_args(
        filename, num_workers=num_workers, chunk_size=chunk_size
    )
    with multiprocessing.Pool(processes=num_workers) as p:
        data = p.starmap(fn, args)

    return itertools.chain.from_iterable(data)


def map_lines(
    filename: Union[str, os.PathLike],
    fn: Callable[[str], Any],
    num_workers: int = multiprocessing.cpu_count(),
    chunk_size: Optional[int] = None,
) -> Iterable:
    args = _get_chunkified_args(
        filename, fn, num_workers=num_workers, chunk_size=chunk_size
    )
    with multiprocessing.Pool(processes=num_workers) as p:
        data = p.starmap(_iter_line_wrapper, args)

    return itertools.chain.from_iterable(data)
