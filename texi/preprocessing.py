# -*- coding: utf-8 -*-

import itertools
import re
import string
import unicodedata
from typing import Callable, Iterable, List, Optional, TypeVar, Union

import torch

LIGATURE_TABLE = {
    42802: "AA",
    42803: "aa",
    198: "AE",
    230: "ae",
    42804: "AO",
    42805: "ao",
    42806: "AU",
    42807: "au",
    42808: "AV",
    42809: "av",
    42810: "AV",
    42811: "av",
    42812: "AY",
    42813: "ay",
    128624: "et",
    64256: "ff",
    64259: "fi",
    64260: "fl",
    64257: "fi",
    64258: "fl",
    338: "OE",
    339: "oe",
    42830: "OO",
    42831: "oo",
    7838: "ſs",
    223: "ſz",
    64262: "st",
    64261: "ſt",
    42792: "TZ",
    42793: "tz",
    7531: "ue",
    42848: "VY",
    42849: "vy",
}


def split(text: str, sep: str) -> List:
    """Split text by separators."""
    sep = re.escape("".join(sep))

    return re.findall(rf"[^{sep}]+[{sep}]?", text)


def remove_control_chars(text: str) -> str:
    """Remove all control characters.

    Be careful that '\n' is will be removed. If you would like to split
    text later, just be careful.
    """

    return "".join(c for c in text if unicodedata.category(c)[0] != "C")


def remove_english_punctuations(text: str) -> str:
    """Remove all English punctuations."""

    return text.translate(str.maketrans("", "", string.punctuation))


def replace_whitesplaces(text: str, replacement: str = " ") -> str:
    """Replace consecutive whitespaces."""

    return re.sub(r"\s+", replacement, text)


def replace_ligatures(text: str) -> str:
    """Replace ligatures with non-ligatures."""

    return text.translate(LIGATURE_TABLE)


def get_opencc(conversion: Optional[str] = "t2s") -> Callable[[str], str]:
    # pylint: disable=import-outside-toplevel
    import opencc

    converter = opencc.OpenCC(conversion)

    def _wrapper(text):
        return converter.convert(text)

    return _wrapper


class LabelEncoder(object):
    T = TypeVar("T", bound="LabelEncoder")

    def __init__(
        self, labels: Optional[Iterable[str]] = None, default: Optional[str] = None
    ) -> None:
        self.default = default
        self.init(labels)

    def __len__(self):
        return len(self.index2label)

    @property
    def labels(self):
        return list(self.label2index)

    @property
    def num_labels(self):
        return len(self)

    @property
    def default_index(self):
        return self.get_index(self.default)

    def reset(self) -> None:
        self.index2label = {}
        self.label2index = {}

        if self.default:
            self.label2index[self.default] = 0
            self.index2label[0] = self.default

    def init(self, labels: Iterable[str]) -> None:
        self.reset()

        if not labels:
            return

        for label in labels:
            self.label2index.setdefault(label, len(self.label2index))

        self.index2label = {v: k for k, v in self.label2index.items()}

    def add(self, label: str) -> int:
        index = self.label2index.setdefault(label, len(self))
        self.index2label.setdefault(index, label)

        return index

    def get_index(self, label: str) -> int:
        index = self.label2index.get(label)
        if index is None:
            if self.default is not None:
                return 0

            raise KeyError(label)

        return index

    def get_label(self, index: int) -> str:
        label = self.index2label.get(index)
        if label is None:
            if self.default is not None:
                return self.default

            raise KeyError(index)

        return label

    def encode_label(
        self, label: str, return_tensors: Optional[str] = None
    ) -> Union[int, torch.Tensor]:
        index = self.get_index(label)

        if return_tensors == "pt":
            return torch.tensor(index, dtype=torch.int64)

        if isinstance(return_tensors, str):
            raise ValueError('`return_tensors` should be "pt" or None')

        return index

    def decode_label(self, index: Union[int, torch.LongTensor]) -> str:
        if isinstance(index, torch.LongTensor):
            if index.ndim > 0:
                raise ValueError(
                    f"tensor should be 0-d tensor, got: ndim == {index.ndim}"
                )
            index = int(index.cpu().numpy())

        if not isinstance(index, int):
            raise ValueError(
                "`index` should be int or torch.LongTensor, "
                f"got: {index.__class__.__name__}"
            )

        return self.get_label(index)

    def encode(
        self, labels: Iterable[str], return_tensors: Optional[str] = None
    ) -> Union[List[int], torch.Tensor]:
        indices = [self.get_index(x) for x in labels]
        if return_tensors == "pt":
            return torch.tensor(indices, dtype=torch.int64)

        if isinstance(return_tensors, str):
            raise ValueError('`return_tensors` should be "pt" or None')

        return indices

    def decode(self, indices: Union[torch.LongTensor, Iterable[int]]) -> List[str]:
        if isinstance(indices, torch.LongTensor):
            if indices.ndim != 1:
                raise ValueError(
                    f"tensor should be 1-d tensor, got: ndim == {indices.ndim}"
                )
            indices = indices.cpu().numpy()

        labels = [self.get_label(x) for x in indices]

        return labels

    @classmethod
    def from_iterable(cls, labels: Iterable[Iterable[str]]) -> T:
        return cls(itertools.chain.from_iterable(labels))
