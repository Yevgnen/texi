# -*- coding: utf-8 -*-

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def _init_builder(type_):
    def __init__(self, *args, **kwargs):
        super(type(self), self).__init__()
        self.encoder = type_(*args, **kwargs)

    return __init__


def forward(
    self,
    inputs,
    lengths,
    batch_first=True,
    enforce_sorted=False,
    padding_value=0.0,
    total_length=None,
):
    # ATM `lengths` must be 1d long tensor on CPU.
    # https://github.com/pytorch/pytorch/issues/43227
    packed = pack_padded_sequence(
        inputs, lengths.cpu(), batch_first=batch_first, enforce_sorted=enforce_sorted
    )
    hidden, states = self.encoder(packed)
    hidden, _ = pad_packed_sequence(
        hidden,
        batch_first=batch_first,
        padding_value=padding_value,
        total_length=total_length,
    )

    return hidden, states


_recurrent_types = {
    "RNN": nn.RNN,
    "GRU": nn.GRU,
    "LSTM": nn.LSTM,
}


def _init():
    for name, type_ in _recurrent_types.items():
        class_ = type(
            name, (nn.Module,), {"__init__": _init_builder(type_), "forward": forward}
        )
        globals()[name] = class_


_init()


def get_rnn(cell: str) -> nn.Module:
    return globals()[cell.upper()]
