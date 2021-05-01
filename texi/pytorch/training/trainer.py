# -*- coding: utf-8 -*-
# pylint: disable=no-self-use

import abc
import enum
import logging
import os
from typing import Callable, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from carton.logger import log_dict
from carton.logger import setup_logger as carton_setup_logger
from carton.random import set_seed
from ignite.engine import Engine, Events
from ignite.handlers import TerminateOnNan
from ignite.metrics import BatchWise, EpochWise, Metric
from ignite.utils import convert_tensor as ignite_convert_tensor
from ignite.utils import setup_logger
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from texi.pytorch.dataset.dataset import Batch, Dataset
from texi.pytorch.metrics import (
    classification_metrics,
    question_answering_metrics,
    ranking_metrics,
    sequence_labeling_metrics,
)
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
from texi.pytorch.utils import get_default_arguments

logger = logging.getLogger(__name__)

MetricGroup = Mapping[str, Metric]
TrainStepFunction = Callable[[Engine, nn.Module, Batch, nn.Module], Dict]
EvalStepFunction = Callable[[Engine, nn.Module, Batch], Dict]
UpdateFunction = Callable[[Engine, Batch], Dict]


def setup_env(params: Params):
    set_seed(params["seed"])

    os.makedirs(os.path.dirname(params["log_file"]), exist_ok=True)
    carton_setup_logger(level=logging.INFO, filename=params["log_file"])


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

    if params.log_steps > 0:
        logger_handlers = setup_logger_handlers(
            params.save_path,
            params.log_steps,
            params.to_dict(),
            trainer,
            net,
            optimizer,
            evaluators,
            tensorboard=params.tensorboard,
            wandb=params.wandb,
            debug=params.debug,
        )
        handlers.update(logger_handlers)
    else:
        logger.warning("Logger handlers not set")

    trainer.add_event_handler(
        Events.EXCEPTION_RAISED,
        build_exception_handler(trainer, evaluate_handlers["test_evaluate_handler"]),
    )

    return handlers


def build_train_step_function(
    net: nn.Module,
    optimizer: Optimizer,
    loss_function: nn.Module,
    train_step: TrainStepFunction,
    device: torch.device = torch.device("cpu"),
    non_blocking: bool = False,
    max_grad_norm: Optional[float] = None,
    gradient_accumulation_steps: Optional[int] = None,
) -> UpdateFunction:
    def _step(engine, batch):
        net.train()
        batch = convert_tensor(batch, device=device, non_blocking=non_blocking)
        output = train_step(engine, net, batch, loss_function)
        loss = output["loss"]
        loss.backward()
        if (
            not gradient_accumulation_steps
            or engine.state.iteration % gradient_accumulation_steps == 0
        ):
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
        engine.state.metrics["loss"] = loss.item()

        return output

    return _step


def build_eval_step_function(
    net: nn.Module,
    eval_step: EvalStepFunction,
    device: torch.device = torch.device("cpu"),
    non_blocking: bool = False,
) -> UpdateFunction:
    def _step(engine, batch):
        # pylint: disable=unused-argument

        net.eval()
        with torch.no_grad():
            batch = convert_tensor(batch, device=device, non_blocking=non_blocking)
            output = eval_step(engine, net, batch)

            return output

    return _step


class Trainer(metaclass=abc.ABCMeta):
    def get_metrics(self, train: bool = True) -> MetricGroup:
        # pylint: disable=unused-argument

        return {}

    def predict_step(self, output: Dict) -> torch.Tensor:
        return output["logit"].argmax(dim=-1)

    def train_step(
        self, _: Engine, net: nn.Module, batch: Batch, loss_function: nn.Module
    ) -> Dict:
        x, y = batch
        logit = net(x)
        loss = loss_function(logit, y)
        output = {"x": x, "y": y, "logit": logit, "loss": loss}
        output["y_pred"] = self.predict_step(output)

        return output

    def eval_step(self, _: Engine, net: nn.Module, batch: Batch) -> Dict:
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
        max_grad_norm: Optional[float] = None,
        gradient_accumulation_steps: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
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
                max_grad_norm=max_grad_norm,
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

        if max_grad_norm is not None:
            logger.info("Max gradient norm set to: %f", max_grad_norm)

        if gradient_accumulation_steps is not None:
            logger.info(
                "Gradient accumulation steps set to: %d", gradient_accumulation_steps
            )

        return engine

    def get_evaluator(
        self,
        net: nn.Module,
        eval_step: Optional[EvalStepFunction] = None,
        metrics: Optional[MetricGroup] = None,
        device: torch.device = torch.device("cpu"),
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
        device: torch.device = torch.device("cpu"),
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
        params: Params,
        net: nn.Module,
        optimizer: Optimizer,
        loss_function: nn.Module,
        train_step: Optional[TrainStepFunction] = None,
        eval_step: Optional[EvalStepFunction] = None,
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

        params = params.to_dict()

        trainer_params = _get_default_arguments(self.get_trainer, params)
        trainer_params.update(train_step=train_step)
        trainer = self.get_trainer(net, optimizer, loss_function, **trainer_params)

        evaluator_params = _get_default_arguments(self.get_evaluators, params)
        evaluator_params.update(eval_step=eval_step)
        evaluators = self.get_evaluators(net, **evaluator_params)

        return trainer, evaluators

    def setup_data_loaders(self, data_loaders):
        self.data_loaders = data_loaders
        for mode, loader in self.data_loaders.items():
            logger.info("Dataset description [%s]:", mode)

            if isinstance(loader.dataset, Dataset):
                stats = loader.dataset.describe()
            else:
                stats = {"size": len(loader.dataset)}

            log_dict(logger, stats)

    def setup(
        self,
        params: Params,
        data_loaders: Mapping[str, DataLoader],
        net: nn.Module,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: Optional[_LRScheduler] = None,
        train_step: Optional[TrainStepFunction] = None,
        eval_step: Optional[EvalStepFunction] = None,
    ) -> None:
        self.trainer, self.evaluators = self.get_engines(
            params, net, optimizer, loss_fn, train_step=train_step, eval_step=eval_step
        )
        self.setup_data_loaders(data_loaders)

        self.net = net
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.handlers = setup_handlers(
            params,
            self.trainer,
            self.evaluators,
            self.data_loaders,
            net,
            optimizer,
            lr_scheduler=lr_scheduler,
        )

        self.params = params

    def run(self, *args, **kwargs) -> None:
        logger.info("Training params:")
        log_dict(logger, self.params)

        self.params.to_yaml()

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
        self, _: Engine, net: nn.Module, batch: Batch, loss_function: nn.Module
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

    def eval_step(self, _: Engine, net: nn.Module, batch: Batch) -> Dict:
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
        self, _: Engine, net: nn.Module, batch: Batch, loss_function: nn.Module
    ) -> Dict:
        train_steps = {
            "pointwise": self._pointwise_step,
            "pairwise": self._pairwise_step,
            "pointwise_pairwise": self._pointwise_pairwise_step,
        }

        return train_steps[self.mode](net, batch, loss_function)

    def eval_step(self, _: Engine, net: nn.Module, batch: Batch) -> Dict:
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
        self, _: Engine, net: nn.Module, batch: Batch, loss_function: nn.Module
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
