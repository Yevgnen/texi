# -*- coding: utf-8 -*-

import torch
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce


class GeneralAccuracy(Metric):
    @reinit__is_reduced
    def reset(self):
        self._num_correct = torch.tensor(0, device=self._device)
        self._num_examples = 0
        super().reset()

    @reinit__is_reduced
    def update(self, output):
        def _detach(t):
            return t.detach() if isinstance(t, torch.Tensor) else t

        y_pred, y = output

        y_pred = _detach(y_pred)
        y = _detach(y)

        correct = [yi_pred == yi for yi_pred, yi in zip(y_pred, y)]

        self._num_correct += sum(correct)
        self._num_examples += len(correct)

    @sync_all_reduce("_num_examples", "_num_correct")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                (
                    "GeneralAccuracy must have at least one example "
                    "before it can be computed."
                )
            )
        return self._num_correct.item() / self._num_examples
