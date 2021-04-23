# -*- coding: utf-8 -*-

from typing import Callable, Dict, Union

import torch
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce

from texi.metrics import prf1
from texi.preprocessing import LabelEncoder


class NerMetrics(Metric):
    def __init__(
        self,
        entity_label_encoder: LabelEncoder,
        negative_entity_index: int,
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
    def update(self, output: Dict) -> None:
        def _combine_span_and_label(y):
            # label: [B, E]
            # span: [B, E, 2]
            # mask: [B, E]
            span_with_label = torch.cat(
                [y["label"].unsqueeze(dim=-1), y["span"]], axis=-1
            )
            negative_mask = ~y["mask"].unsqueeze(dim=-1).bool()
            span_with_label.masked_fill_(negative_mask, -1)

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

        y_pred, y = output

        target = _combine_span_and_label(y)
        prediction = _combine_span_and_label(y_pred)

        targets = _to_tuples(target)
        predictions = _to_tuples(prediction)

        _update(targets, predictions)

    @sync_all_reduce("entity_stat:SUM", "typed_entity_stat:SUM")
    def compute(self) -> Dict[str, float]:
        metrics = prf1(self.entity_stat[0], self.entity_stat[1], self.entity_stat[2])
        typed_metrics = {
            self.entity_label_encoder.decode_label(i): prf1(
                self.typed_entity_stat[i][0],
                self.typed_entity_stat[i][1],
                self.typed_entity_stat[i][2],
            )
            for i in range(len(self.entity_label_encoder))
        }

        return {"all": metrics, **typed_metrics}