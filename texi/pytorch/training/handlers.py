# -*- coding: utf-8 -*-

import logging
import os
import traceback
from typing import Any, Callable, Dict, Mapping, Optional, Union, cast

import torch
import torch.nn as nn
from ignite.contrib.engines.common import (
    add_early_stopping_by_val_score,
    save_best_model_by_val_score,
    setup_wandb_logging,
)
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.base_logger import BaseLogger
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data.dataloader import DataLoader

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
        engine.logger.info("Dataset [train] switched to train mode.")


def build_exception_handler(
    trainer: Engine, test_evaluate_handler: Optional[Callable[[Engine], None]]
) -> Callable[[Engine, Exception], None]:
    def handle_exceptions(engine, e):
        if isinstance(e, KeyboardInterrupt):
            engine.logger.info("KeyboardInterrupt caught. Exiting gracefully.")
            trainer.terminate()

            if callable(test_evaluate_handler):
                test_evaluate_handler(trainer)
        else:
            traceback.print_exc()

    return handle_exceptions


def build_evaluate_handler(
    evaluator: Engine,
    mode: str,
    model: nn.Module,
    dataflow: DataLoader,
    best_model_handler: Optional[Callable[[Engine], None]] = None,
):
    def evaluate_handler(_):
        evaluator.logger.info("Evaluate on [%s]", mode)
        if best_model_handler is not None:
            if best_model_handler.last_checkpoint is not None:
                checkpoint = os.path.join(
                    best_model_handler.save_handler.dirname,
                    best_model_handler.last_checkpoint,
                )
                evaluator.logger.info(
                    "Loading checkpoint %r before evaluate",
                    checkpoint,
                )
                model.load_state_dict(torch.load(checkpoint))

        if isinstance(dataflow.dataset, Dataset):
            dataflow.dataset.eval()
            evaluator.logger.info("Dataset [%s] switched to eval mode.", mode)

        evaluator.run(dataflow)

        evaluator.logger.info("Evaluate metrics [%s]", mode)
        for key, metric in sorted(evaluator.state.metrics.items(), key=lambda x: x[0]):
            # Ignote Dict metrics flattened by ignite.
            if isinstance(metric, Mapping):
                continue

            evaluator.logger.info("%s = %s", key, metric)

    return evaluate_handler


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
        trainer.logger.warning("LR scheduler not set")


def setup_early_stopping_handler(
    params: Params, trainer: Engine, evaluator: Engine
) -> Union[EarlyStopping, None]:
    handler = None
    if params.early_stopping:
        if params.eval_metric is None or params.patience is None:
            raise ValueError(
                "`eval_metric` and `patience` must set when `early_stopping` is set"
            )
        if params.num_save_models < 0:
            trainer.logger.warning("Early stopping is set, but best model is not saved")

        handler = add_early_stopping_by_val_score(
            params.patience,
            evaluator,
            trainer,
            params.eval_metric,
        )
        trainer.logger.info(
            "Early stopping is set with `eval_metric` = %s and `patience` = %d",
            params.eval_metric,
            params.patience,
        )

    return handler


def setup_evaluate_handlers(
    params: Params,
    trainer: Engine,
    evaluators: Mapping[str, Engine],
    model: nn.Module,
    dataflows: Mapping[str, DataLoader],
) -> Dict[str, Any]:
    handlers = dict.fromkeys(
        [
            "best_model_handler",
            "early_stopping_handler",
            "train_evaluate_handler",
            "val_evaluate_handler",
            "test_evaluate_handler",
        ]
    )

    if params.eval_steps == "epoch" or params.eval_steps > 0:
        for mode in ["train", "val"]:
            if mode == "train" and not params.eval_train:
                continue

            evaluate_handler = build_evaluate_handler(
                evaluators[mode], mode, model, dataflows[mode]
            )
            trainer.add_event_handler(get_event(params.eval_steps), evaluate_handler)
            handlers[f"{mode}_evaluate_handler"] = evaluate_handler

            trainer.logger.info("Setup evaluator for [%s]", mode)

        if params.num_save_models > 0:
            handlers["best_model_handler"] = save_best_model_by_val_score(
                params.save_path,
                evaluators["val"],
                model,
                params.eval_metric,
                n_saved=params.num_save_models,
                trainer=trainer,
            )
            trainer.logger.info(
                "Save best model hander set with `num_save_models` = %d",
                params.num_save_models,
            )
        else:
            trainer.logger.warning("Save best model handler not set")

        handlers["early_stopping_handler"] = setup_early_stopping_handler(
            params, trainer, evaluators["val"]
        )
    else:
        trainer.logger.warning("Evaluate handlers not set")
        if params.early_stopping:
            raise ValueError(
                "Evaluate handlers must set when `early_stopping` is set"
                ", check `eval_steps`, `eval_metric` and `patience`"
            )

    test_evaluator = evaluators.get("test")
    if test_evaluator is not None:
        test_loader = dataflows.get("test")
        if test_loader is None:
            raise ValueError(
                "`test_loader` must not be None when `test_evaluator` is passed"
            )

        evaluate_handler = build_evaluate_handler(
            test_evaluator, "test", model, test_loader, handlers["best_model_handler"]
        )
        trainer.add_event_handler(Events.COMPLETED, evaluate_handler)
        handlers["test_evaluate_handler"] = evaluate_handler

        trainer.logger.info("Setup evaluator for [test]")
    else:
        trainer.logger.warning("Test evaluate handlers not set")

    return handlers


def setup_logger_handlers(
    params: Params,
    trainer: Engine,
    model: nn.Module,
    optimizers: Optional[Union[Optimizer, Mapping[str, Optimizer]]] = None,
    evaluators: Optional[Mapping[str, Engine]] = None,
) -> Dict[str, BaseLogger]:
    handlers = dict.fromkeys(["tensorboard_logger", "wandb_logger"])

    if params.log_steps < 0:
        trainer.logger.warning("Logger handlers not set")

        return handlers

    if params.tensorboard:
        tensorboard_dir = os.path.join(params.save_path, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        tensorboard_logger = setup_tb_logging(
            tensorboard_dir,
            trainer,
            optimizers=optimizers,
            evaluators=evaluators,
            log_steps=params.log_steps,
            model=model,
            include_weights_and_grads=params.debug,
        )
        handlers["tensorboard_logger"] = tensorboard_logger

    if params.wandb:
        # FIXME: Duplicated code with evaluator setup.
        # https://github.com/pytorch/ignite/issues/1476#issuecomment-826317167
        def filter_metrics(engine):
            engine.state.metrics = {
                k: v for k, v in engine.state.metrics.items() if not isinstance(v, dict)
            }

        if evaluators:
            for evaluator in evaluators.values():
                evaluator.add_event_handler(Events.COMPLETED, filter_metrics)

        wandb_dir = os.path.join(params.save_path, "wandb")
        os.makedirs(wandb_dir, exist_ok=True)
        wandb_logger = setup_wandb_logging(
            trainer,
            optimizers=optimizers,
            evaluators=evaluators,
            log_every_iters=params.log_steps,
            dir=wandb_dir,
            config=params.to_dict(),
            project=params.project,
            reinit=True,
        )
        wandb_logger.attach(
            trainer, lambda *args, **kwargs: wandb_logger.close, Events.COMPLETED
        )
        if params.debug:
            wandb_logger.watch(model, log="all", log_steps=params.log_steps)
        handlers["wandb_logger"] = wandb_logger

    return handlers
