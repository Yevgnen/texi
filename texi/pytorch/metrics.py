# -*- coding: utf-8 -*-

import itertools
from typing import Callable, Dict, Optional, Sequence, Union

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
from texi.metrics import confusion_matrix, prf1


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


class NerMetrics(Metric):
    def __init__(
        self,
        entity_label_encoder,
        negative_entity_index,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        self.entity_label_encoder = entity_label_encoder
        self.negative_entity_index = negative_entity_index

        super().__init__(output_transform, device=device)

    @reinit__is_reduced
    def reset(self) -> None:
        # TP, FP, FN
        self.entity_stat = torch.zeros((3,), device=self._device)
        self.typed_entity_stat = torch.zeros(
            (len(self.entity_label_encoder), 3), device=self._device
        )

    @reinit__is_reduced
    def update(self, outputs: Dict) -> None:
        def _combine_span_and_label(y):
            # label: [B, R]
            # span: [B, R, 2]
            # mask: [B, R]
            span_with_label = torch.cat(
                [y["label"].unsqueeze(dim=-1), y["span"]], axis=-1
            )
            span_with_label.masked_fill(y["mask"].unsqueeze(dim=-1).bool(), -1)

            return span_with_label

        def _to_tuples(span_with_label):
            span_with_labels = span_with_label.detach().cpu().numpy().tolist()

            tuples = []
            for sample_span_with_labels in span_with_labels:
                sample_tuples = []
                for label, start, end in sample_span_with_labels:
                    negative = label < 0 and start < 0 and end < 0
                    positive = label >= 0 and start >= 0 and end >= 0

                    assert (
                        positive or negative
                    ), f"Unexpected entity span with label: {(label, start, end)}"

                    if label >= 0 and start >= 0 and end >= 0:
                        if label != self.negative_entity_index:
                            sample_tuples += [(label, start, end)]
                tuples += [sample_tuples]

            return tuples

        def _update(all_targets, all_predictions):
            for targets, predictions in zip(all_targets, all_predictions):
                targets, predictions = set(targets), set(predictions)

                for entity in targets & predictions:
                    self.entity_stat[0] += 1
                    self.typed_entity_stat[entity[0]][0] += 1

                for entity in predictions - targets:
                    self.entity_stat[1] += 1
                    self.typed_entity_stat[entity[0]][1] += 1

                for entity in targets - predictions:
                    self.entity_stat[2] += 1
                    self.typed_entity_stat[entity[0]][2] += 1

        y_pred, y = outputs

        target = _combine_span_and_label(y)
        prediction = _combine_span_and_label(y_pred)

        targets = _to_tuples(target)
        predictions = _to_tuples(prediction)

        _update(targets, predictions)

    @sync_all_reduce("entity_stat:SUM", "typed_entity_stat:SUM")
    def compute(self) -> Dict[str, float]:
        def _compute_with_entities():
            metrics = prf1(
                self.entity_stat[0], self.entity_stat[1], self.entity_stat[2]
            )
            typed_metrics = {
                self.entity_label_encoder.decode_label(i): prf1(
                    self.typed_entity_stat[i][0],
                    self.typed_entity_stat[i][1],
                    self.typed_entity_stat[i][2],
                )
                for i in range(len(self.entity_label_encoder))
            }
            return {**metrics, **typed_metrics}

        output = {**_compute_with_entities()}

        return output


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
        metrics[f"top{k}_accuracy"] = TopKCategoricalAccuracy(
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
