# -*- coding: utf-8 -*-
# pylint: disable=not-callable

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

from ignite.contrib.engines import common as logging_base
from ignite.contrib.handlers.tensorboard_logger import (
    GradsHistHandler,
    GradsScalarHandler,
    WeightsHistHandler,
    WeightsScalarHandler,
)
from ignite.engine import Events

if TYPE_CHECKING:
    # pylint: disable=ungrouped-imports
    import torch.nn as nn
    from ignite.contrib.handlers import TensorboardLogger
    from ignite.engine import Engine
    from torch.optim import Optimizer

    from texi.pytorch.training.trainer import Trainer


def setup_tb_logging(
    log_dir: str,
    trainer: Trainer,
    optimizers: Dict[str, Optimizer] = None,
    evaluators: Dict[str, Engine] = None,
    log_steps: int = 1,
    net: Optional[nn.Module] = None,
    include_weights_and_grads: bool = True,
    **kwargs,
) -> TensorboardLogger:
    tb_logger = logging_base.setup_tb_logging(
        log_dir,
        trainer,
        optimizers=optimizers,
        evaluators=evaluators,
        log_every_iters=log_steps,
        **kwargs,
    )

    if include_weights_and_grads:
        tb_logger.attach(
            trainer,
            log_handler=WeightsScalarHandler(net),
            event_name=Events.ITERATION_COMPLETED(every=log_steps),
        )

        tb_logger.attach(
            trainer,
            log_handler=WeightsHistHandler(net),
            event_name=Events.ITERATION_COMPLETED(every=log_steps),
        )

        tb_logger.attach(
            trainer,
            log_handler=GradsScalarHandler(net),
            event_name=Events.ITERATION_COMPLETED(every=log_steps),
        )

        tb_logger.attach(
            trainer,
            log_handler=GradsHistHandler(net),
            event_name=Events.ITERATION_COMPLETED(every=log_steps),
        )

    tb_logger.attach(trainer, lambda *args, **kwargs: tb_logger.close, Events.COMPLETED)

    return tb_logger
