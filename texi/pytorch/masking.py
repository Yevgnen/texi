# -*- coding: utf-8 -*-

from typing import List, Optional, Union

import torch


def create_span_mask(
    starts: Union[List[int], torch.Tensor],
    ends: Union[List[int], torch.Tensor],
    length: int,
    dtype: torch.dtype = torch.int64,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    if len(starts) != len(ends):
        raise ValueError(
            f"`start` and `end` should have same lengths: {len(starts)} != {len(ends)}"
        )

    if isinstance(starts, torch.Tensor):
        if starts.ndim != 1:
            raise ValueError(
                f"`starts` must be 1d if passed as tensor, got ndim == {starts.ndim}"
            )
        starts = starts.tolist()

    if isinstance(ends, torch.Tensor):
        if ends.ndim != 1:
            raise ValueError(
                f"`ends` must be 1d if passed as tensor, got ndim == {ends.ndim}"
            )
        ends = ends.tolist()

    if len(starts) == 0:
        return torch.zeros((0, length), dtype=dtype, device=device)

    start = torch.tensor(starts, dtype=dtype, device=device)
    end = torch.tensor(ends, dtype=dtype, device=device)
    mask = torch.arange(length, dtype=dtype, device=device).unsqueeze(dim=-1)
    mask = (start <= mask) & (mask < end)
    mask = mask.transpose(0, 1).type_as(start)

    return mask


def length_to_mask(
    length: torch.Tensor,
    max_len: Optional[Union[int, torch.Tensor]] = None,
    batch_first: bool = False,
    dtype: torch.dtype = torch.int64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if max_len is None:
        max_len = length.max()

    if device is None:
        device = length.device

    mask = torch.arange(max_len, device=device).unsqueeze(dim=1).expand(
        max_len, len(length)
    ) < length.unsqueeze(dim=0)
    mask = mask.type(dtype)

    return mask.transpose(0, 1) if batch_first else mask


def mask_to_length(
    mask: torch.Tensor,
    batch_first: bool = False,
    dtype: torch.dtype = torch.int64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    length = mask.sum(dim=int(batch_first)).type(dtype)
    if device is not None:
        length = length.to(device)

    return length
