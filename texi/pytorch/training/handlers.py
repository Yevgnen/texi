# -*- coding: utf-8 -*-

import logging
import os
from typing import Dict, Mapping, Optional, Union, cast

import torch.nn as nn
from ignite.contrib.engines.common import setup_wandb_logging
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.base_logger import BaseLogger
from ignite.engine import Engine, Events
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from texi.pytorch.dataset.dataset import Dataset
from texi.pytorch.logger import setup_tb_logging
from texi.pytorch.training.params import Params

logger = logging.getLogger(__name__)


def get_event(steps: Union[int, str]) -> Events:
    if isinstance(steps, int):
        return Events.ITERATION_COMPLETED(every=steps)

    if steps == "epoch":
        return Events.EPOCH_COMPLETED(every=1)

    raise ValueError(
        (
            "Event frequency should be either"
            " integer for Events.ITERATION_COMPLETED"
            ' or "epoch" for Events.EPOCH_COMPLETED'
        )
    )


def handle_dataset_mode(engine: Engine) -> None:
    if isinstance(engine.state.dataloader.dataset, Dataset):
        engine.state.dataloader.dataset.train()
        logger.info("Dataset [train] switched to train mode.")


def setup_progress_bar(
    params: Params, trainer: Engine, evaluators: Mapping[str, Engine]
) -> None:
    if params.pbar_steps > 0:
        ProgressBar(ncols=0).attach(
            trainer,
            metric_names="all",
            event_name=Events.ITERATION_COMPLETED(every=params.pbar_steps),
        )

        for evaluator in evaluators.values():
            ProgressBar(ncols=0).attach(
                evaluator,
                event_name=Events.ITERATION_COMPLETED(every=params.pbar_steps),
            )


def setup_lr_scheduler(
    params: Params, trainer: Engine, lr_scheduler: _LRScheduler
) -> None:
    if lr_scheduler is not None:
        if params.schedule_steps == "epoch" or params.schedule_steps > 0:
            trainer.add_event_handler(
                get_event(params.schedule_steps),
                lambda engine: cast(_LRScheduler, lr_scheduler).step(),
            )
        else:
            raise ValueError(
                '`schedule_steps` must be positve or "epoch"'
                " when `lr_scheduler` is passed"
            )

    else:
        logger.warning("LR scheduler not set")


def setup_logger_handlers(
    save_path: str,
    log_steps: int,
    params: Mapping,
    trainer: Engine,
    net: nn.Module,
    optimizers: Optional[Union[Optimizer, Mapping[str, Optimizer]]] = None,
    evaluators: Optional[Mapping[str, Engine]] = None,
    tensorboard: bool = False,
    wandb: bool = False,
    debug: bool = False,
) -> Dict[str, BaseLogger]:
    handlers = {}

    if tensorboard:
        tensorboard_dir = os.path.join(save_path, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        tensorboard_logger = setup_tb_logging(
            tensorboard_dir,
            trainer,
            optimizers=optimizers,
            evaluators=evaluators,
            log_steps=log_steps,
            net=net,
            include_weights_and_grads=debug,
        )
        handlers["tensorboard_logger"] = tensorboard_logger

    if wandb:
        # FIXME: Duplicated code with evaluator setup.
        # https://github.com/pytorch/ignite/issues/1476#issuecomment-826317167
        def filter_metrics(engine):
            engine.state.metrics = {
                k: v for k, v in engine.state.metrics.items() if not isinstance(v, dict)
            }

        if evaluators:
            for evaluator in evaluators.values():
                evaluator.add_event_handler(Events.COMPLETED, filter_metrics)

        wandb_dir = os.path.join(save_path, "wandb")
        os.makedirs(wandb_dir, exist_ok=True)
        wandb_logger = setup_wandb_logging(
            trainer,
            optimizers=optimizers,
            evaluators=evaluators,
            log_every_iters=log_steps,
            dir=wandb_dir,
            config=params,
            project=params["project"],
            reinit=True,
        )
        wandb_logger.attach(
            trainer, lambda *args, **kwargs: wandb_logger.close, Events.COMPLETED
        )
        if debug:
            wandb_logger.watch(net, log="all", log_steps=log_steps)
        handlers["wandb_logger"] = wandb_logger

    return handlers
