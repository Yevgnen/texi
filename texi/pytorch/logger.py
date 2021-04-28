# -*- coding: utf-8 -*-
# pylint: disable=not-callable


from ignite.contrib.engines import common as logging_base
from ignite.contrib.handlers.tensorboard_logger import (
    GradsHistHandler,
    GradsScalarHandler,
    WeightsHistHandler,
    WeightsScalarHandler,
)
from ignite.engine import Events


def setup_tb_logging(
    log_dir,
    trainer,
    optimizers=None,
    evaluators=None,
    log_steps=1,
    net=None,
    include_weights_and_grads=True,
    **kwargs,
):
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
