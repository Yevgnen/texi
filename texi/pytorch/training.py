# -*- coding: utf-8 -*-
# pylint: disable=no-self-use

import abc
import enum
import logging
import os
import pprint
import traceback
from typing import Callable, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from ignite.contrib.engines.common import (
    add_early_stopping_by_val_score,
    save_best_model_by_val_score,
    setup_wandb_logging,
)
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, EarlyStopping, TerminateOnNan
from ignite.metrics import BatchWise, EpochWise, Metric
from ignite.utils import convert_tensor as ignite_convert_tensor
from ignite.utils import setup_logger
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from texi.pytorch.dataset.dataset import Batch
from texi.pytorch.logger import setup_tb_logging
from texi.pytorch.metrics import (
    classification_metrics,
    question_answering_metrics,
    ranking_metrics,
    sequence_labeling_metrics,
)
from texi.pytorch.optim import optim
from texi.pytorch.utils import get_default_arguments

logger = logging.getLogger(__name__)

MetricGroup = Mapping[str, Metric]
TrainStepFunction = Callable[[nn.Module, Batch, nn.Module], Dict]
EvalStepFunction = Callable[[nn.Module, Batch], Dict]
UpdateFunction = Callable[[Engine, Batch], Dict]


def convert_tensor(*args, **kwargs):
    return ignite_convert_tensor(*args, **kwargs)


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
    metrics: Optional[MetricGroup] = None,
    train: bool = True,
) -> Engine:
    engine.logger = setup_logger(name, filepath=log_file)

    if metrics:
        for metric_name, metric in metrics.items():
            metric.attach(
                engine, metric_name, usage=BatchWise() if train else EpochWise()
            )
    return engine


def setup_save_handlers(
    trainer: Engine,
    evaluator: Engine,
    net: nn.Module,
    eval_metric: Optional[str] = None,
    save_path: str = ".",
    patience: int = 5,
) -> Tuple[Checkpoint, EarlyStopping]:
    best_model_handler = None
    early_stopping_handler = None
    if eval_metric:
        best_model_handler = save_best_model_by_val_score(
            save_path, evaluator, net, eval_metric, trainer=trainer
        )
        if patience > 0:
            early_stopping_handler = add_early_stopping_by_val_score(
                patience, evaluator, trainer, eval_metric
            )

    return best_model_handler, early_stopping_handler


def setup_handlers(
    trainer: Engine,
    train_evaluator: Engine,
    val_evaluator: Engine,
    train_loader: DataLoader,
    val_loader: DataLoader,
    net: nn.Module,
    optimizer: Optimizer,
    eval_metric: str,
    save_path: str,
    lr_scheduler: Optional[_LRScheduler] = None,
    log_freq: int = 1,
    eval_freq: int = 1,
    schedule_freq: int = 1,
    patience: int = 10,
    eval_event_name: Events = Events.EPOCH_COMPLETED,
    schedule_event_name: Events = Events.EPOCH_COMPLETED,
    test_evaluator: Optional[Engine] = None,
    test_loader: Optional[DataLoader] = None,
    tensorboard: bool = True,
    wandb: bool = True,
    watch: bool = False,
    params: Optional[Mapping] = None,
) -> Dict:
    # pylint: disable=not-callable, unused-argument, unused-variable
    # pylint: disable=too-many-arguments, too-many-locals

    # Other event handlers.
    def step_schedulers(engine):
        lr_scheduler.step()
        engine.logger.info(
            "Learning rate updated: %s",
            ", ".join(str(x) for x in lr_scheduler.get_last_lr()),
        )

    def build_evaluate_handler(
        dataset, evaluator, data_loader, best_model_handler=None
    ):
        def evaluate_handler(engine):
            evaluator.logger.info("Evaluate on [%s]", dataset)
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
                    net.load_state_dict(torch.load(checkpoint))
            evaluator.run(data_loader)
            evaluator.logger.info(pprint.pformat(evaluator.state.metrics))

        return evaluate_handler

    def handle_exceptions(engine, e):
        if isinstance(e, KeyboardInterrupt):
            engine.logger.info("User terminated")
            trainer.terminate()
            test_evaluate_handler(trainer)
        else:
            traceback.print_exc()

    # Setup handlers.
    handlers = {}
    ProgressBar(ncols=0).attach(
        trainer,
        metric_names="all",
        event_name=Events.ITERATION_COMPLETED(every=log_freq),
    )

    trainer.add_event_handler(Events.EXCEPTION_RAISED, handle_exceptions)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    if lr_scheduler is not None:
        trainer.add_event_handler(
            schedule_event_name(every=schedule_freq), step_schedulers
        )

    trainer.add_event_handler(
        eval_event_name(every=eval_freq),
        build_evaluate_handler("train", train_evaluator, train_loader),
    )

    trainer.add_event_handler(
        eval_event_name(every=eval_freq),
        build_evaluate_handler("val", val_evaluator, val_loader),
    )

    best_model_handler, early_stopping_handler = setup_save_handlers(
        trainer,
        val_evaluator,
        net,
        eval_metric,
        save_path,
        patience,
    )
    handlers["best_model_handler"] = best_model_handler
    handlers["early_stopping_handler"] = early_stopping_handler

    if test_evaluator is not None:
        test_evaluate_handler = build_evaluate_handler(
            "test", test_evaluator, test_loader, best_model_handler
        )
        trainer.add_event_handler(Events.COMPLETED, test_evaluate_handler)

    # Setup loggers.
    evaluators = {"validating/train": train_evaluator, "validating/val": val_evaluator}
    if test_evaluator is not None:
        evaluators.update({"testing": test_evaluator})

    for evaluator in evaluators.values():
        ProgressBar(ncols=0).attach(
            evaluator,
            event_name=Events.ITERATION_COMPLETED(every=log_freq),
        )

    log_dir = os.path.join(save_path, "log/")
    os.makedirs(log_dir, exist_ok=True)
    if tensorboard:
        tensorboard_logger = setup_tb_logging(
            log_dir,
            trainer,
            optimizers=optimizer,
            evaluators=evaluators,
            log_freq=log_freq,
            net=net,
            include_weights_and_grads=watch,
        )
        handlers["tensorboard_logger"] = tensorboard_logger

    if wandb:
        wandb_logger = setup_wandb_logging(
            trainer,
            optimizers=optimizer,
            evaluators=evaluators,
            log_every_iters=log_freq,
            dir=log_dir,
            config=params,
            project=params["project"],
            name=params["test_name"],
            reinit=True,
        )
        wandb_logger.attach(
            trainer, lambda *args, **kwargs: wandb_logger.close, Events.COMPLETED
        )
        if watch:
            wandb_logger.watch(net, log="all", log_freq=log_freq)
        handlers["wandb_logger"] = wandb_logger

    return handlers


def build_train_step_function(
    net: nn.Module,
    optimizer: Optimizer,
    loss_function: nn.Module,
    train_step: TrainStepFunction,
    device: torch.device = "cpu",
    non_blocking: bool = False,
    clip_grad_norm: Optional[int] = None,
    gradient_accumulation_steps: Optional[int] = None,
) -> UpdateFunction:
    def _step(engine, batch):
        net.train()
        batch = convert_tensor(batch, device=device, non_blocking=non_blocking)
        output = train_step(net, batch, loss_function)
        loss = output["loss"]
        loss.backward()
        if (
            not gradient_accumulation_steps
            or engine.state.iteration % gradient_accumulation_steps == 0
        ):
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
        engine.state.metrics["loss"] = loss.item()

        return output

    return _step


def build_eval_step_function(
    net: nn.Module,
    eval_step: EvalStepFunction,
    device: torch.device = "cpu",
    non_blocking: bool = False,
) -> UpdateFunction:
    def _step(engine, batch):
        # pylint: disable=unused-argument

        net.eval()
        with torch.no_grad():
            batch = convert_tensor(batch, device=device, non_blocking=non_blocking)
            output = eval_step(net, batch)

            return output

    return _step


class Trainer(metaclass=abc.ABCMeta):
    def get_metrics(self, train: bool = True) -> MetricGroup:
        # pylint: disable=unused-argument

        return {}

    def predict_step(self, output: Dict) -> torch.Tensor:
        return output["logit"].argmax(dim=-1)

    def train_step(
        self, net: nn.Module, batch: Batch, loss_function: nn.Module
    ) -> Dict:
        x, y = batch
        logit = net(x)
        loss = loss_function(logit, y)
        output = {"x": x, "y": y, "logit": logit, "loss": loss}
        output["y_pred"] = self.predict_step(output)

        return output

    def eval_step(self, net: nn.Module, batch: Batch) -> Dict:
        x, y = batch
        logit = net(x)
        output = {"x": x, "y": y, "logit": logit}
        output["y_pred"] = self.predict_step(output)

        return output

    def get_trainer(
        self,
        net: nn.Module,
        optimizer: Optimizer,
        loss_function: nn.Module,
        train_step: Optional[TrainStepFunction] = None,
        clip_grad_norm: Optional[int] = None,
        gradient_accumulation_steps: Optional[int] = None,
        device: torch.device = "cpu",
        non_blocking: bool = False,
        metrics: Optional[MetricGroup] = None,
        log_file: Optional[str] = None,
        name: str = "trainer",
    ) -> Engine:
        # pylint: disable=too-many-arguments

        engine = Engine(
            build_train_step_function(
                net,
                optimizer,
                loss_function,
                train_step or self.train_step,
                device=device,
                non_blocking=non_blocking,
                clip_grad_norm=clip_grad_norm,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )
        )

        setup_engine(
            engine,
            name,
            log_file=log_file,
            metrics=metrics or self.get_metrics(train=True),
            train=True,
        )

        return engine

    def get_evaluator(
        self,
        net: nn.Module,
        eval_step: Optional[EvalStepFunction] = None,
        metrics: Optional[MetricGroup] = None,
        device: torch.device = "cpu",
        non_blocking: bool = False,
        log_file: Optional[str] = None,
        name: str = "evaluator",
    ) -> Engine:
        engine = Engine(
            build_eval_step_function(
                net,
                eval_step or self.eval_step,
                device=device,
                non_blocking=non_blocking,
            )
        )

        setup_engine(
            engine,
            name,
            log_file=log_file,
            metrics=metrics or self.get_metrics(train=False),
            train=False,
        )

        return engine

    def get_evaluators(
        self,
        net: nn.Module,
        eval_step: EvalStepFunction = None,
        metrics: Optional[MetricGroup] = None,
        device: torch.device = "cpu",
        non_blocking: bool = False,
        log_file: Optional[str] = None,
    ) -> Dict[str, Engine]:
        evaluators = {
            name: self.get_evaluator(
                net,
                eval_step=eval_step,
                metrics=metrics,
                device=device,
                non_blocking=non_blocking,
                log_file=log_file,
                name=name,
            )
            for name in ["train_evaluator", "val_evaluator", "test_evaluator"]
        }

        return evaluators

    def get_engines(
        self,
        net: nn.Module,
        optimizer: Optimizer,
        loss_function: nn.Module,
        train_step: Optional[TrainStepFunction] = None,
        eval_step: Optional[EvalStepFunction] = None,
        params: Optional[Mapping] = None,
    ) -> Tuple[Engine, Dict[str, Engine]]:
        # Try to get engine params. Can't pass **params to
        # `self.get_trainer` or `self.get_evaluators` because `params`
        # contains invalid parameters for them.
        def _get_default_arguments(f, params):
            default_arguments = get_default_arguments(f)
            for key, value in params.items():
                if key in default_arguments:
                    default_arguments[key] = value

            return default_arguments

        if not params:
            params = {}

        trainer_params = _get_default_arguments(self.get_trainer, params)
        trainer_params.update(train_step=train_step)
        trainer = self.get_trainer(net, optimizer, loss_function, **trainer_params)

        evaluator_params = _get_default_arguments(self.get_evaluators, params)
        evaluator_params.update(eval_step=eval_step)
        evaluators = self.get_evaluators(net, **evaluator_params)

        return trainer, evaluators

    def setup(
        self,
        data_loaders: Mapping[str, DataLoader],
        net: nn.Module,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: Optional[_LRScheduler] = None,
        train_step: Optional[TrainStepFunction] = None,
        eval_step: Optional[EvalStepFunction] = None,
        params: Optional[Mapping] = None,
    ) -> None:
        if not params:
            raise ValueError("`params` must not be None")

        self.trainer, self.evaluators = self.get_engines(
            net,
            optimizer,
            loss_fn,
            train_step=train_step,
            eval_step=eval_step,
            params=params,
        )
        self.data_loaders = data_loaders

        self.net = net
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.params = params

        setup_handlers(
            self.trainer,
            self.evaluators["train_evaluator"],
            self.evaluators["val_evaluator"],
            self.data_loaders["train"],
            self.data_loaders["val"],
            net,
            optimizer,
            params["eval_metric"],
            params["save_path"],
            lr_scheduler=lr_scheduler,
            eval_freq=params["eval_freq"],
            log_freq=params["log_freq"],
            schedule_freq=params["schedule_freq"],
            patience=params["patience"],
            test_evaluator=self.evaluators["test_evaluator"],
            test_loader=self.data_loaders["test"],
            tensorboard=params["tensorboard"],
            wandb=params["wandb"],
            params=params,
        )

    def run(self, *args, **kwargs) -> None:
        logger.info("Start training with params:")
        logger.info(pprint.pformat(self.params))

        kwargs.setdefault("max_epochs", self.params["max_epochs"])
        self.trainer.run(self.data_loaders["train"], *args, **kwargs)


class TrainerForClassification(Trainer):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def get_metrics(self, train: bool = True) -> MetricGroup:
        metrics = classification_metrics(
            lambda x: ((x["logit"] if self.num_classes > 2 else x["y_pred"]), x["y"]),
            train=train,
        )

        return metrics

    def train_step(
        self, net: nn.Module, batch: Batch, loss_function: nn.Module
    ) -> Dict:
        x, y = batch
        logit = net(x)
        loss = loss_function(
            logit.squeeze(dim=1), y.long() if self.num_classes > 2 else y.float()
        )
        if self.num_classes == 2:
            y_pred = torch.sigmoid(logit).round().type_as(y)
        else:
            y_pred = logit.argmax(dim=1).type_as(y)

        return {"x": x, "y": y, "y_pred": y_pred, "logit": logit, "loss": loss}

    def eval_step(self, net: nn.Module, batch: Batch) -> Dict:
        x, y = batch
        logit = net(batch)
        logit = logit.squeeze(dim=1)
        if self.num_classes == 2:
            y_pred = torch.sigmoid(logit).round().type_as(y)
        else:
            y_pred = logit.argmax(dim=1).type_as(y)

        return {"x": x, "y": y, "y_pred": y_pred, "logit": logit}


class TrainerForRanking(Trainer):
    class Mode(str, enum.Enum):
        POINTWISE = "pointwise"
        PAIRWISE = "pairwise"
        POINTWISE_PAIRWISE = "pointwise_pairwise"

    def __init__(self, mode: "TrainerForRanking.Mode"):
        super().__init__()
        self.mode = mode

    def _pointwise_step(self, net, batch, loss_function):
        x, y = batch
        logit = net(x)
        loss = loss_function(logit, y.float())
        y_pred = torch.sigmoid(logit).round().type_as(y)

        return {"x": x, "y": y, "y_pred": y_pred, "logit": logit, "loss": loss}

    def _pairwise_step(self, net, batch, loss_function):
        x, y = batch
        logit = net(x)
        loss = loss_function(*logit.chunk(2))
        y_pred = torch.sigmoid(logit).round().type_as(y)

        return {"x": x, "y": y, "y_pred": y_pred, "logit": logit, "loss": loss}

    def _pointwise_pairwise_step(self, net, batch, loss_function):
        x, y = batch
        logit = net(x)
        loss = 0.5 * loss_function[0](logit, y.float())
        loss += 0.5 * loss_function[1](*logit.chunk(2))
        y_pred = torch.sigmoid(logit).round().type_as(y)

        return {"x": x, "y": y, "y_pred": y_pred, "logit": logit, "loss": loss}

    def get_metrics(self, train: bool = True) -> MetricGroup:
        metrics = ranking_metrics(lambda x: (x["y_pred"], x["y"]), train=train)

        return metrics

    def train_step(
        self, net: nn.Module, batch: Batch, loss_function: nn.Module
    ) -> Dict:
        train_steps = {
            "pointwise": self._pointwise_step,
            "pairwise": self._pairwise_step,
            "pointwise_pairwise": self._pointwise_pairwise_step,
        }

        return train_steps[self.mode](net, batch, loss_function)

    def eval_step(self, net: nn.Module, batch: Batch) -> Dict:
        y, y_pred = [], []
        for sample_x, sample_y in batch:
            logit = net(sample_x)
            y_pred += [logit.transpose(0, 1)]
            y += [sample_y.argmax(dim=0)]
        y_pred = torch.cat(y_pred, dim=0)
        y = torch.stack(y, dim=0)

        return {"x": batch[0], "y": y, "y_pred": y_pred}


class TrainerForSequenceLabeling(Trainer):
    def __init__(self, decoder, labels):
        super().__init__()
        self.decoder = decoder
        self.labels = labels

    def get_metrics(self, train: bool = True) -> MetricGroup:
        def _output_transform(output):
            x, y, logit = output["x"], output["y"], output["logit"]
            _, y = self.decoder((x, y))
            x, y_pred = self.decoder((x, logit))

            return {"x": x, "y": y, "y_pred": y_pred}

        metrics = sequence_labeling_metrics(_output_transform, self.labels, train=train)

        return metrics

    def train_step(
        self, net: nn.Module, batch: Batch, loss_function: nn.Module
    ) -> Dict:
        x, y = batch
        logit = net(x)
        loss = loss_function(logit, y, x["tag_mask"])
        output = {"x": x, "y": y, "logit": logit, "loss": loss}
        output["y_pred"] = self.predict_step(output)

        return output


class TrainerForQuestionAnaswering(Trainer):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def get_metrics(self, train: bool = True) -> MetricGroup:
        metrics = question_answering_metrics(
            lambda x: (x["y_pred"], x["y"]), train=train
        )

        return metrics

    def predict_step(self, output: Dict) -> torch.Tensor:
        return self.decoder(output["logit"])
