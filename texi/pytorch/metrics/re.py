# -*- coding: utf-8 -*-

from typing import Callable, Dict, Union

import torch
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce

from texi.metrics import prf1
from texi.preprocessing import LabelEncoder


class ReMetrics(Metric):
    def __init__(
        self,
        relation_label_encoder: LabelEncoder,
        negative_relation_index: int,
        relation_filter_threshold: float,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        self.relation_label_encoder = relation_label_encoder
        self.negative_relation_index = negative_relation_index
        self.relation_filter_threshold = relation_filter_threshold

        super().__init__(output_transform, device=device)

    @reinit__is_reduced
    def reset(self) -> None:
        # TP, FP, FN
        self.relation_stat = torch.zeros((3,), device=self._device)
        self.typed_relation_stat = torch.zeros(
            (len(self.relation_label_encoder), 3), device=self._device
        )

    @reinit__is_reduced
    def update(self, output: Dict) -> None:
        def _combine_pair_and_label(y):
            label = (y["label"] > self.relation_filter_threshold).long()

            pair_with_one_hot_label = torch.cat(
                [
                    label.unsqueeze(dim=-1),
                    y["pair"]
                    .unsqueeze(dim=-2)
                    .repeat(1, 1, len(self.relation_label_encoder), 1),
                ],
                axis=-1,
            )
            negative_mask = ~y["mask"].view(*y["mask"].size(), 1, 1).bool()
            pair_with_one_hot_label.masked_fill_(negative_mask, -1)

            return pair_with_one_hot_label  # [B, R, R']

        def _to_tuples(pair_with_one_hot_label, entity_spans):
            pair_with_one_hot_labels = (
                pair_with_one_hot_label.detach().cpu().numpy().tolist()
            )
            entity_spans = entity_spans.detach().cpu().numpy().tolist()

            tuples = []
            for sample_pair_with_one_hot_labels, sample_entity_spans in zip(
                pair_with_one_hot_labels, entity_spans
            ):
                sample_tuples = []
                for pair_with_labels in sample_pair_with_one_hot_labels:
                    for k, (label, head, tail) in enumerate(pair_with_labels):
                        negative = label < 0 and head < 0 and tail < 0
                        positive = label >= 0 and head >= 0 and tail >= 0

                        assert (
                            positive or negative
                        ), f"Unexpected relation pair with label: {(label, head, tail)}"

                        # Need to map `head` and `tail` to corresponding
                        # spans because they may have different indices
                        # in target and prediction.
                        head_span = tuple(sample_entity_spans[head])
                        tail_span = tuple(sample_entity_spans[tail])

                        if label > 0 and k != self.negative_relation_index:
                            sample_tuples += [(k, head_span, tail_span, label)]

                tuples += [sample_tuples]

            return tuples

        def _update(all_targets, all_predictions):
            for targets, predictions in zip(all_targets, all_predictions):
                targets, predictions = set(targets), set(predictions)

                for relation in targets & predictions:
                    self.relation_stat[0] += 1
                    self.typed_relation_stat[relation[0]][0] += 1

                for relation in predictions - targets:
                    self.relation_stat[1] += 1
                    self.typed_relation_stat[relation[0]][1] += 1

                for relation in targets - predictions:
                    self.relation_stat[2] += 1
                    self.typed_relation_stat[relation[0]][2] += 1

        y_pred, y = output

        target = _combine_pair_and_label(y)
        prediction = _combine_pair_and_label(y_pred)

        targets = _to_tuples(target, y["entity_span"])
        predictions = _to_tuples(prediction, y_pred["entity_span"])

        _update(targets, predictions)

    @sync_all_reduce("relation_stat:SUM", "typed_relation_stat:SUM")
    def compute(self) -> Dict[str, float]:
        metrics = prf1(
            self.relation_stat[0], self.relation_stat[1], self.relation_stat[2]
        )
        typed_metrics = {
            self.relation_label_encoder.decode_label(i): prf1(
                self.typed_relation_stat[i][0],
                self.typed_relation_stat[i][1],
                self.typed_relation_stat[i][2],
            )
            for i in range(len(self.relation_label_encoder))
        }

        return {"re_all": metrics, **typed_metrics}
