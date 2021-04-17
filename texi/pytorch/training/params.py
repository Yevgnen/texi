# -*- coding: utf-8 -*-

import argparse
import copy
import os
from typing import Dict, Optional

from carton.params import Params as ParamDict
from carton.random import random_state

from texi.pytorch.utils import device


class Params(object):
    # pylint: disable=too-many-instance-attributes
    def __init__(self, **kwargs):
        # Project
        self.project = kwargs.get("project")
        if not self.project:
            raise ValueError("`project` must not be empty")

        # Datasets
        self.val_size = kwargs.get("val_size", 0.1)
        self.test_size = kwargs.get("test_size", 0.2)

        # Training
        self.seed = random_state(kwargs.get("seed"))
        self.save_path = kwargs.get("save_path", "output/")
        self.log_file = os.path.join(self.save_path, "output.log")
        self.max_epochs = kwargs.get("max_epochs")
        self.lr = kwargs.get("lr")
        if self.lr is None:
            raise ValueError("`lr` must not be None")
        self.lr_warmup = kwargs.get("lr_warmup")
        self.weight_decay = kwargs.get("weight_decay")
        self.max_grad_norm = kwargs.get("max_grad_norm")
        self.schedule_steps = kwargs.get("schedule_steps", -1)

        # Dataloader
        self.train_batch_size = kwargs.get("train_batch_size", 32)
        self.eval_batch_size = kwargs.get("eval_batch_size", 32)
        self.device = device(kwargs.get("device", "cuda"))
        self.pin_memory = kwargs.get("pin_memory", True)
        self.non_blocking = kwargs.get("non_blocking", True)

        # Evaluation & Early Stopping
        self.eval_train = kwargs.get("eval_train", False)
        self.eval_steps = kwargs.get("eval_steps", -1)
        self.early_stopping = kwargs.get("early_stopping", False)
        self.eval_metric = kwargs.get("eval_metric")
        self.patience = kwargs.get("patience")

        # Logging
        self.log_steps = kwargs.get("log_steps", -1)
        self.tensorboard = kwargs.get("tensorboard", False)
        self.wandb = kwargs.get("wandb", False)
        self.debug = kwargs.get("debug", False)

    def __repr__(self):
        return repr(self.__dict__)

    def to_dict(self) -> Dict:
        return copy.deepcopy(self.__dict__)

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