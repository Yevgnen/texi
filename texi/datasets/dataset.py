# -*- coding: utf-8 -*-

from __future__ import annotations

import abc
import itertools
import json
import os
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Generic, Optional, Sequence, Type, TypeVar, Union, cast

import torch
import tqdm
from ignite.utils import convert_tensor

from texi.utils import ModeKeys, PhaseMixin

T_co = TypeVar("T_co", covariant=True)


class DatasetTransformMixin(Iterable, metaclass=abc.ABCMeta):
    _mixin_attributes: list[str] = []
    _mixin_transform: Optional[str] = None
    _mixin_inverse_transform: Optional[str] = None
    __iter__: Callable
    __getitem__: Callable

    def _check_transform(self):
        if any(hasattr(self, x) for x in self._mixin_attributes):
            raise RuntimeError(f"Can not call `.{self._mixin_transform}()` twice.")

    def _check_inverse_transform(self):
        if any(not hasattr(self, x) for x in self._mixin_attributes):
            raise RuntimeError(
                f"Can not call `.{self._mixin_inverse_transform}()`"
                f" before `.{self._mixin_transform}()`."
            )

    def _remove_attributes(self):
        for attr in self._mixin_attributes:
            delattr(self, attr)


class SplitableMixin(DatasetTransformMixin):
    _mixin_attributes = ["_split_lengths"]
    _mixin_transform = "split"
    _mixin_inverse_transform = "merge"

    def split(self, fn: Callable) -> None:
        self._check_transform()

        splits = [fn(x) for x in self]
        lengths = [len(x) for x in splits]

        self._split_lengths = lengths

        self.examples = list(itertools.chain.from_iterable(splits))

    def merge(self, fn: Callable) -> None:
        self._check_inverse_transform()

        examples = []

        offset = 0
        for length in self._split_lengths:
            examples += [fn(self[offset : offset + length])]
            offset += length

        self.examples = examples

        self._remove_attributes()


class MaskableMixin(DatasetTransformMixin):
    _mixin_attributes = [
        "_masked_positives",
        "_masked_negatives",
    ]
    _mixin_transform = "mask"
    _mixin_inverse_transform = "unmask"

    def mask(self, fn: Callable) -> None:
        self._check_transform()

        positives, negatives = [], []
        for i, example in enumerate(self):
            flag = fn(example)
            if flag:
                positives += [(i, example)]
            else:
                negatives += [(i, example)]

        self._masked_positives = positives
        self._masked_negatives = negatives

        self.examples = [x[1] for x in positives]

    def unmask(self) -> None:
        self._check_inverse_transform()

        examples = sorted(
            self._masked_positives + self._masked_negatives, key=lambda x: x[0]
        )

        self.examples = [x[1] for x in examples]

        self._remove_attributes()


class EagerEncodeMixin(DatasetTransformMixin):
    _mixin_attributes = ["_original_examples"]
    _mixin_transform = "eager_encode"
    _mixin_inverse_transform = "eager_decode"
    device: Optional[torch.device]
    encode_batch: Callable
    collate_train: Callable
    collate_eval: Callable
    is_train: Callable

    def eager_encode(self) -> None:
        if hasattr(self, "_original_examples"):
            examples = self._original_examples  # type: ignore
        else:
            examples = self.examples  # type: ignore
            self._original_examples = self.examples  # type: ignore

        encoded = self.encode_batch(
            tqdm.tqdm(examples, desc="Encode batch:", ncols=0, leave=False)
        )  # type: ignore

        if self.device is not None:
            encoded = convert_tensor(encoded, device=self.device, non_blocking=True)

        self.examples = encoded

    def eager_decode(self) -> None:
        self._check_inverse_transform()
        self.examples = self._original_examples  # type: ignore

        self._remove_attributes()

    def collate_fn(self, batch: Sequence) -> Any:
        if not hasattr(self, "_original_examples"):
            return super().collate_fn(batch)

        fn = self.collate_train if self.is_train() else self.collate_eval
        collated = fn(batch)

        return collated


class Dataset(PhaseMixin, MaskableMixin, SplitableMixin, Generic[T_co]):
    T = TypeVar("T", bound="Dataset")

    def __init__(
        self,
        examples: Union[Iterable[T_co], Callable[[], Iterable[T_co]]],
        mode: ModeKeys = ModeKeys.TRAIN,
        device: Optional[torch.device] = None,
    ) -> None:
        if callable(examples):
            self.load_examples = examples  # type: Optional[Callable]
            self.examples = None  # type: ignore
        else:
            self.examples = list(examples)
            self.load_examples = None

        self.mode = mode
        self.device = device

    def __getitem__(self, key) -> T_co:
        self._check_loaded()

        return self.examples[key]

    def __iter__(self):
        if self.examples is not None:
            yield from iter(self.examples)
        elif callable(self.load_examples):
            yield from self.load_examples()

    def __len__(self):
        self._check_loaded()

        return len(self.examples)

    def __repr__(self):
        if self.examples is None:
            return f"{self.__class__.__name__}(Not loaded)"

        return f"{self.__class__.__name__}({len(self)} examples)"

    def _check_loaded(self):
        if self.examples is None:
            raise RuntimeError("Dataset is not loaded, call `.load()` first")

    def load(self: T) -> T:
        if callable(self.load_examples) and self.examples is None:
            self.examples = list(self.load_examples())

        return self

    def map(self, fn: Callable[..., Union[T_co, Sequence[T_co]]]) -> None:
        self._check_loaded()

        examples = [fn(x) for x in cast(list, self.examples)]
        if examples and isinstance(examples[0], list):
            examples = list(itertools.chain.from_iterable(examples))  # type: ignore

        self.examples = examples

    def describe(self) -> dict[str, Any]:
        return {"size": len(self)}

    def encode(self, example) -> Any:  # pylint: disable=no-self-use
        return example

    def encode_batch(self, batch: Sequence) -> list:
        return list(map(self.encode, batch))

    def collate_train(self, batch: Sequence) -> Any:
        raise NotImplementedError()

    def collate_eval(self, batch: Sequence) -> Any:
        return self.collate_train(batch)

    def collate_fn(self, batch: Sequence) -> Any:
        encoded = self.encode_batch(batch)

        if self.device is not None:
            encoded = convert_tensor(encoded, device=self.device, non_blocking=True)

        fn = self.collate_train if self.is_train() else self.collate_eval
        collated = fn(encoded)

        return collated

    @classmethod
    def from_json_iter(
        cls: Type[T],
        filename: Union[str, os.PathLike],
        format_function: Optional[Callable] = lambda x: x,
        array: bool = False,
        mode: ModeKeys = ModeKeys.TRAIN,
        device: Optional[torch.device] = None,
    ) -> T:
        def _iter_whole_file():
            with open(filename) as f:
                yield from map(format_function, json.load(f))

        def _iter_multiple_lines():
            with open(filename) as f:
                for line in f:
                    line = line.rstrip()
                    if line:
                        yield format_function(json.loads(line))

        fn = _iter_whole_file if array else _iter_multiple_lines

        return cls(fn, mode=mode, device=device)

    @classmethod
    def from_json(
        cls: Type[T],
        filename: Union[str, os.PathLike],
        mode: ModeKeys = ModeKeys.TRAIN,
        device: Optional[torch.device] = None,
    ) -> T:
        return cls(list(cls.from_json_iter(filename)), mode=mode, device=device)


class Datasets(Generic[T_co]):
    T = TypeVar("T", bound="Datasets")

    def __init__(
        self,
        train: Optional[Union[Dataset[T_co], Iterable, Callable]] = None,
        val: Optional[Union[Dataset[T_co], Iterable, Callable]] = None,
        test: Optional[Union[Dataset[T_co], Iterable, Callable]] = None,
        dirname: Optional[Union[str, os.PathLike]] = None,
        filename: Optional[Union[str, os.PathLike]] = None,
    ) -> None:
        def _wrap(d):
            if not isinstance(d, Dataset):
                return Dataset(d)

            return d

        self.train = _wrap(train)
        self.val = _wrap(val)
        self.test = _wrap(test)
        self.dirname = dirname
        self.filename = filename

        self.modes = {"train", "val", "test"}

    def __getitem__(self, key) -> Dataset[T_co]:
        assert key in self.modes

        return getattr(self, key)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(train={self.train}"
            f", val={self.val}, test={self.test}"
            f", dirname={self.dirname}, filename={self.filename})"
        )

    def _map_dataset_methods(self, method, *args, **kwargs):
        outputs = dict.fromkeys(self.modes)
        for mode, dataset in self.items():
            if dataset is not None:
                outputs[mode] = getattr(dataset, method)(*args, **kwargs)

        return outputs

    def load(self: T) -> T:
        for mode in self.modes:
            dataset = getattr(self, mode)
            if dataset is not None:
                dataset.load()

        return self

    def items(self) -> Iterable[tuple[str, Dataset[T_co]]]:
        for mode in self.modes:
            yield mode, getattr(self, mode)

    def to_dict(self) -> dict[str, Optional[Dataset[T_co]]]:
        return {
            "train": self.train,
            "val": self.val,
            "test": self.test,
        }

    def map(self, fn: Callable[[T_co], Any]) -> None:
        self._map_dataset_methods("map", fn)

    def split(self, fn: Callable[[T_co], Any]) -> None:
        self._map_dataset_methods("split", fn)

    def mask(self, fn: Callable[[T_co], Any]) -> None:
        self._map_dataset_methods("mask", fn)

    @staticmethod
    def _map_modekeys(mode):
        return {
            "train": ModeKeys.TRAIN,
            "val": ModeKeys.EVAL,
            "test": ModeKeys.PREDICT,
        }[mode]

    @classmethod
    def from_dir(
        cls: Type[T],
        dirname: Union[str, os.PathLike],
        device: Optional[torch.device] = None,
    ) -> T:
        raise NotImplementedError()


class JSONDatasets(Datasets):
    T = TypeVar("T", bound="JSONDatasets")

    files = {
        "train": "train.json",
        "val": "val.json",
        "test": "test.json",
    }

    @classmethod
    def format(cls: Type[T], x: Any) -> Any:
        return x

    @classmethod
    def from_dir(
        cls: Type[T],
        dirname: Union[str, os.PathLike],
        device: Optional[torch.device] = None,
        array: bool = False,
    ) -> T:
        # pylint: disable=arguments-differ

        data = {
            key: Dataset.from_json_iter(
                os.path.join(dirname, value),
                cls.format,
                array=array,
                mode=cls._map_modekeys(key),
                device=device,
            )
            for key, value in cls.files.items()
        }  # type: dict[str, Dataset[dict]]

        return cls(train=data["train"], val=data["val"], test=data["test"])
