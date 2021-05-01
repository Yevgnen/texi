# -*- coding: utf-8 -*-

import logging
import os
from typing import Callable, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from carton.logger import log_dict
from carton.logger import setup_logger as carton_setup_logger
from carton.random import set_seed
from ignite.contrib.engines.common import setup_common_training_handlers
from ignite.engine import Engine, Events
from ignite.handlers import DiskSaver, TerminateOnNan
from ignite.metrics import BatchWise, EpochWise, Metric
from ignite.utils import convert_tensor, setup_logger
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from texi.pytorch.dataset.dataset import Batch, Dataset
from texi.pytorch.optim import optim
from texi.pytorch.training.handlers import (
    build_exception_handler,
    handle_dataset_mode,
    setup_evaluate_handlers,
    setup_logger_handlers,
    setup_lr_scheduler,
    setup_progress_bar,
)
from texi.pytorch.training.params import Params

logger = logging.getLogger(__name__)

Dataflows = Mapping[str, DataLoader]
Metrics = Mapping[str, Metric]
TrainStepFunction = Callable[[Engine, nn.Module, Batch, nn.Module], Dict]
EvalStepFunction = Callable[[Engine, nn.Module, Batch], Dict]


def setup_env(params: Params):
    set_seed(params["seed"])

    os.makedirs(os.path.dirname(params["log_file"]), exist_ok=True)
    carton_setup_logger(level=logging.INFO, filename=params["log_file"])


def configure_optimizers(
    net: nn.Module,
    params: Mapping,
) -> Tuple[Optimizer, _LRScheduler]:
    optimizer_params = {k: v for k, v in params.items() if k != "lr_scheduler"}
    optimizer = optim.optim(net.parameters(), **optimizer_params)

    lr_scheduler = None
    lr_scheduler_params = params.get("lr_scheduler")
    if lr_scheduler_params:
        lr_scheduler = optim.lr_scheduler(optimizer, **lr_scheduler_params)

    return optimizer, lr_scheduler


def setup_engine(
    engine: Engine,
    name: str,
    log_file: Optional[str] = None,
    metrics: Optional[Metrics] = None,
    train: bool = True,
) -> Engine:
    engine.logger = setup_logger(name, filepath=log_file)

    if metrics:
        for metric_name, metric in metrics.items():
            metric.attach(
                engine, metric_name, usage=BatchWise() if train else EpochWise()
            )
    return engine


def create_trainer(
    train_step: TrainStepFunction,
    params: Params,
    model: nn.Module,
    criteria: nn.Module,
    optimizer: Optimizer,
    lr_scheduler: Optional[_LRScheduler] = None,
    metrics: Optional[Metrics] = None,
    name: str = "trainer",
    with_handlers: bool = True,
) -> Engine:
    def step(engine, batch):
        model.train()
        batch = convert_tensor(
            batch, device=params.device, non_blocking=params.non_blocking
        )
        output = train_step(engine, model, batch, criteria)
        loss = output["loss"]
        loss.backward()
        if (
            not params.gradient_accumulation_steps
            or engine.state.iteration % params.gradient_accumulation_steps == 0
        ):
            if params.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), params.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
        engine.state.metrics["loss"] = loss.item()

        return output

    trainer = Engine(step)
    trainer = setup_engine(
        trainer, name, log_file=params.log_file, metrics=metrics, train=True
    )

    if with_handlers:
        to_save = {
            "trainer": trainer,
            "model": model,
            "optimizer": optimizer,
        }
        if lr_scheduler is not None:
            to_save.update({"lr_scheduler": lr_scheduler})

        trainer.add_event_handler(Events.EPOCH_STARTED, handle_dataset_mode)
        setup_lr_scheduler(params, trainer, lr_scheduler)

        setup_common_training_handlers(
            trainer,
            train_sampler=None,  # TODO
            to_save=to_save,
            save_every_iters=params.save_steps,
            lr_scheduler=lr_scheduler,
            with_gpu_stats=False,
            output_names=["loss"],
            with_pbars=True,
            with_pbar_on_iters=True,
            log_every_iters=params.pbar_steps,
            stop_on_nan=True,
            clear_cuda_cache=True,
            save_handler=DiskSaver(params.save_path, require_empty=False),
        )

    return trainer


def create_evaluator(
    eval_step: EvalStepFunction,
    params: Params,
    model: nn.Module,
    metrics: Metrics,
    tag: str,
) -> Engine:
    @torch.no_grad()
    def step(engine, batch):
        model.eval()
        batch = convert_tensor(
            batch, device=params.device, non_blocking=params.non_blocking
        )
        output = eval_step(engine, model, batch)

        return output

    evaluator = Engine(step)
    evaluator = setup_engine(
        evaluator,
        f"evaluator/{tag}",
        log_file=params.log_file,
        metrics=metrics,
        train=False,
    )

    return evaluator


def create_evaluators(
    eval_step: EvalStepFunction, params: Params, model: nn.Module, metrics: Metrics
) -> Dict[str, Engine]:
    return {
        mode: create_evaluator(eval_step, params, model, metrics, tag=mode)
        for mode in ["train", "val", "test"]
    }


def setup_handlers(
    params: Params,
    trainer: Engine,
    evaluators: Mapping[str, Engine],
    data_loaders: Mapping[str, DataLoader],
    net: nn.Module,
    optimizer: Optimizer,
    lr_scheduler: Optional[_LRScheduler] = None,
) -> Dict:
    # pylint: disable=not-callable, unused-argument, unused-variable
    # pylint: disable=too-many-locals, too-many-arguments

    # Setup general handlers.
    handlers = {}
    trainer.add_event_handler(Events.EPOCH_STARTED, handle_dataset_mode)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    # Setup progress bars.
    setup_progress_bar(params, trainer, evaluators)

    # Setup lr scheduler.
    setup_lr_scheduler(params, trainer, lr_scheduler)

    # Setup evaluate handlers.
    evaluate_handlers = setup_evaluate_handlers(
        params, trainer, evaluators, net, data_loaders
    )
    handlers.update(evaluate_handlers)

    # Setup logger handlers.
    logger_handlers = setup_logger_handlers(params, trainer, net, optimizer, evaluators)
    handlers.update(logger_handlers)

    trainer.add_event_handler(
        Events.EXCEPTION_RAISED,
        build_exception_handler(trainer, evaluate_handlers["test_evaluate_handler"]),
    )

    return handlers


def describe_dataflows(dataflows, logger_: Optional[logging.Logger] = None) -> None:
    if logger_ is None:
        logger_ = logger

    for mode, flow in dataflows.items():
        logger_.info("Dataset description [%s]:", mode)

        if isinstance(flow.dataset, Dataset):
            stats = flow.dataset.describe()
        else:
            stats = {"size": len(flow.dataset)}

        log_dict(logger_, stats)


class Env(object):
    def get_metrics(self, train: bool):
        pass

    def train_step(
        self, engine: Engine, model: nn.Module, batch: Batch, criteria: nn.Module
    ) -> Dict:
        pass

    def eval_step(self, engine: Engine, model: nn.Module, batch: Batch) -> Dict:
        pass

    def setup(
        self,
        params: Params,
        dataflows: Dataflows,
        model: nn.Module,
        criteria: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: Optional[_LRScheduler] = None,
    ):
        self.dataflows = dataflows

        self.trainer = create_trainer(
            self.train_step,
            params,
            model,
            criteria,
            optimizer,
            lr_scheduler=lr_scheduler,
            metrics=self.get_metrics(train=True),
        )

        self.evaluators = create_evaluators(
            self.eval_step, params, model, self.get_metrics(train=False)
        )

        evaluate_handlers = setup_evaluate_handlers(
            params, self.trainer, self.evaluators, model, dataflows
        )
        logger_handlers = setup_logger_handlers(
            params, self.trainer, model, optimizer, self.evaluators
        )
        handlers = {**evaluate_handlers, **logger_handlers}

        describe_dataflows(dataflows, self.trainer.logger)

        return handlers
