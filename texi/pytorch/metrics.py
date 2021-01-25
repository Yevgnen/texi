# -*- coding: utf-8 -*-

import itertools
from typing import Callable, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from ignite.exceptions import NotComputableError
from ignite.metrics import (
    Accuracy,
    Fbeta,
    Metric,
    Precision,
    Recall,
    TopKCategoricalAccuracy,
)
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce

from texi.apps.ner import conlleval
from texi.metrics import confusion_matrix


class GeneralAccuracy(Metric):
    def __init__(self, output_transform=lambda x: x, device="cpu"):
        self._num_correct = None
        self._num_examples = None
        super().__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._num_correct = torch.tensor(0, device=self._device)
        self._num_examples = 0
        super().reset()

    @reinit__is_reduced
    def update(self, output):
        if isinstance(output, dict):
            y_pred, y = output["y_pred"], output["y"]
        else:
            y_pred, y = output

        correct = [yi_pred == yi for yi_pred, yi in zip(y_pred, y)]

        self._num_correct += torch.sum(correct).to(self._device)
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


class MeanReciprocalRank(Metric):
    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__(output_transform, device=device)

    @reinit__is_reduced
    def reset(self) -> None:
        self._ranks = []

    @reinit__is_reduced
    def update(self, output: Sequence) -> None:
        y_pred, y = output
        if y.dim() == 1:
            y = F.one_hot(y, num_classes=y_pred.size()[1])
        ranks = y_pred.argsort(dim=-1, descending=True).argsort(dim=-1)
        rank = ranks.masked_select(y > 0).float()
        self._ranks += [1 / (rank + 1)]

    @sync_all_reduce("_ranks")
    def compute(self) -> Union[float, torch.Tensor]:
        if self._num_examples == 0:
            raise NotComputableError(
                "MeanReciprocalRank must have at"
                "least one example before it can be computed."
            )

        return torch.mean(torch.cat(self._ranks, dim=0)).item()


class SequenceLabelingMetrics(Metric):
    _required_output_keys = ("x", "y", "y_pred")

    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        labels: Optional[Sequence[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__(output_transform, device=device)
        self.labels = list(labels)

    @reinit__is_reduced
    def reset(self) -> None:
        self._x = []
        self._y = []
        self._y_pred = []

    @reinit__is_reduced
    def update(self, output: Sequence) -> None:
        x, y, y_pred = output
        self._x += x
        self._y += y
        self._y_pred += y_pred

    @sync_all_reduce("_x", "_y_pred", "_y")
    def compute(self) -> Union[float, torch.Tensor]:
        if not self._x:
            raise NotComputableError(
                "SequenceLabelingMetrics must have at"
                "least one example before it can be computed."
            )

        output = conlleval(zip(self._x, self._y, self._y_pred))
        metrics = {**output["metrics"]}
        for tag, tag_metrics in output["tags"].items():
            for tag_metric, value in tag_metrics.items():
                metrics["_".join([tag, tag_metric])] = value

        y = [*itertools.chain.from_iterable(self._y)]
        y_pred = [*itertools.chain.from_iterable(self._y_pred)]

        metrics["confusion"] = confusion_matrix(y, y_pred, labels=self.labels)

        return metrics


def classification_metrics(output_transform, train=True):
    if train:
        return {"accuracy": Accuracy(output_transform=output_transform)}

    metrics = {}
    precision = Precision(output_transform=output_transform, average=False)
    recall = Recall(output_transform=output_transform, average=False)
    metrics["accuracy"] = Accuracy(output_transform=output_transform)
    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["f1"] = Fbeta(
        1.0,
        average=False,
        precision=precision,
        recall=recall,
    )

    return metrics


def ranking_metrics(output_transform, train=True):
    if train:
        return {}

    metrics = {
        "mean_reciprocal_rank": MeanReciprocalRank(output_transform=output_transform)
    }
    for k in [1, 3, 5]:
        metrics["top{k}_accuracy".format(k=k)] = TopKCategoricalAccuracy(
            k=k, output_transform=output_transform
        )

    return metrics


def sequence_labeling_metrics(output_transform, labels, train=True):
    if train:
        return {}

    def _output_transform_for_confusion_matrix(output):
        x, y, logits = output
        output_y, output_y_pred = [], []
        for length, yi, y_predi in zip(x["length"], y, logits):
            output_y += [yi[:length]]
            output_y_pred += [y_predi[:length]]
        output_y = torch.cat(output_y, dim=0)
        output_y_pred = torch.cat(output_y_pred, dim=0)

        return output_y_pred, output_y

    metrics = {
        "sequence_labeling_metrics": SequenceLabelingMetrics(output_transform, labels),
    }

    return metrics


def question_answering_metrics(output_transform, train=True):
    if train:
        return {}

    metrics = {}

    return metrics
