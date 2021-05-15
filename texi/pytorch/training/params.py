# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import datetime
import os
from collections.abc import ItemsView, KeysView, ValuesView
from typing import Optional, Type, TypeVar

from carton.params import Params as ParamDict
from carton.random import random_state

from texi.pytorch.utils import device


class Params(object):
    # pylint: disable=too-many-instance-attributes

    T = TypeVar("T", bound="Params")

    def __init__(self, **kwargs):
        # Project
        self.project = kwargs.get("project")
        if not self.project:
            raise ValueError("`project` must not be empty")

        # Datasets
        self.data_dir = kwargs.get("data_dir")
        self.train_data = kwargs.get("train_data")
        self.val_data = kwargs.get("val_data")
        self.test_data = kwargs.get("test_data")
        self.val_size = kwargs.get("val_size", 0.1)
        self.test_size = kwargs.get("test_size", 0.2)

        # Training
        self.backend = kwargs.get("backend")
        self.nproc_per_node = kwargs.get("nproc_per_node")
        self.seed = random_state(kwargs.get("seed"))
        self.save_path = os.path.join(
            kwargs.get("save_path", "output/"), datetime.datetime.now().isoformat()
        )
        self.log_file = os.path.join(self.save_path, "output.log")
        self.max_epochs = kwargs.get("max_epochs")
        self.lr = kwargs.get("lr")
        if self.lr is None:
            raise ValueError("`lr` must not be None")
        self.lr_warmup = kwargs.get("lr_warmup")
        self.weight_decay = kwargs.get("weight_decay")
        self.max_grad_norm = kwargs.get("max_grad_norm")
        self.schedule_steps = kwargs.get("schedule_steps", -1)
        self.gradient_accumulation_steps = kwargs.get("gradient_accumulation_steps")

        # Dataloader
        self.train_batch_size = kwargs.get("train_batch_size", 32)
        self.eval_batch_size = kwargs.get("eval_batch_size", 32)
        self.device = device(kwargs.get("device", "cuda"))
        self.pin_memory = kwargs.get("pin_memory", True)
        self.non_blocking = kwargs.get("non_blocking", True)
        self.num_workers = kwargs.get("num_workers", 0)
        self.eager_encode = kwargs.get("eager_encode", True)

        # Checkpoint
        self.save_steps = kwargs.get("save_steps", 1000)

        # Evaluation
        self.eval_train = kwargs.get("eval_train", False)
        self.eval_steps = kwargs.get("eval_steps", -1)
        self.eval_metric = kwargs.get("eval_metric")
        self.save_best_models = kwargs.get(
            "save_best_models",
            1
            if (self.eval_steps == "epoch" or self.eval_steps > 0)
            and self.eval_metric is not None
            else 0,
        )

        # Early Stopping
        self.early_stopping = kwargs.get("early_stopping", False)
        self.patience = kwargs.get("patience")

        # Logging
        self.pbar_steps = kwargs.get("pbar_steps", 1)
        self.log_steps = kwargs.get("log_steps", -1)
        self.log_level = kwargs.get("log_level", "INFO")
        self.tensorboard = kwargs.get("tensorboard", False)
        self.wandb = kwargs.get("wandb", False)
        self.debug = kwargs.get("debug", False)
        self.sample_dir = kwargs.get(
            "sample_dir", os.path.join(self.save_path, "samples")
        )

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return repr(self.__dict__)

    def to_dict(self) -> dict:
        return copy.deepcopy(self.__dict__)

    def keys(self) -> KeysView:
        return self.__dict__.keys()

    def values(self) -> ValuesView:
        return self.__dict__.values()

    def items(self) -> ItemsView:
        return self.__dict__.items()

    def to_yaml(self, filename: Optional[str] = None) -> None:
        if not filename:
            filename = os.path.join(self.save_path, "params.yaml")

        ParamDict(**self.to_dict()).to_yaml(filename)

    def to_json(self, filename: Optional[str] = None) -> None:
        if not filename:
            filename = os.path.join(self.save_path, "params.json")

        ParamDict(**self.to_dict()).to_json(filename)

    def to_toml(self, filename: Optional[str] = None) -> None:
        if not filename:
            filename = os.path.join(self.save_path, "params.toml")

        ParamDict(**self.to_dict()).to_toml(filename)

    @classmethod
    def from_yaml(cls: Type[T], filename: str) -> T:
        return cls(**ParamDict.from_yaml(filename))

    @classmethod
    def from_json(cls: Type[T], filename: str) -> T:
        return cls(**ParamDict.from_json(filename))

    @classmethod
    def from_toml(cls: Type[T], filename: str) -> T:
        return cls(**ParamDict.from_toml(filename))


def main():
    # pylint: disable=import-outside-toplevel

    def parse_args():
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument(
            "--output",
            type=str,
            default="template.yaml",
            help="Output template file name.",
        )

        return parser.parse_args()  # pylint: disable=redefined-outer-name

    args = parse_args()
    params = ParamDict(**Params(project="NOT SET", lr="NOT SET").to_dict())
    _, ext = os.path.splitext(args.output)
    exts = {".yaml", ".json", ".toml"}
    if ext not in exts:
        raise RuntimeError(
            (f"Unsupported file type: {ext}, supported file types are: {exts}")
        )

    getattr(params, f"to_{ext[1:]}")(args.output)


if __name__ == "__main__":
    main()
