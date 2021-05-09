# -*- coding: utf-8 -*-

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Union

import torch
from carton.collections import flatten_dict
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
        prefix: str = "",
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        self.relation_label_encoder = relation_label_encoder
        self.negative_relation_index = negative_relation_index
        self.relation_filter_threshold = relation_filter_threshold
        self.prefix = prefix

        super().__init__(output_transform, device=device)

    @reinit__is_reduced
    def reset(self) -> None:
        # TP, FP, FN
        self.tpfpfn = torch.zeros((3,), device=self._device)
        self.typed_tpfpfn = torch.zeros(
            (len(self.relation_label_encoder), 3), device=self._device
        )

    @reinit__is_reduced
    def update(self, output: tuple[Mapping, Mapping]) -> None:
        def _expand_entities(y):
            # Expand head/tail index by corresponding entity type and span.

            # label: [B, R, R']
            # pair: [B, R, 2]
            # mask: [B, R]
            label = (y["label"] > self.relation_filter_threshold).long()

            batch_size = y["label"].size(0)
            indices = torch.arange(batch_size).view(batch_size, 1, 1)

            # entity_span: [B, R, 4]
            # entity_label: [B, R, 2]
            # entity: [B, R, 6]
            entity_span = y["entity_span"][indices, y["pair"]].flatten(start_dim=-2)
            entity_label = y["entity_label"][indices, y["pair"]]
            entity = torch.cat([entity_span, entity_label], dim=-1)

            return {
                "label": label,
                "entity": entity,
                "mask": y["mask"],
            }

        def _update(y, y_pred, stat, index=None):
            # Compare head/tail entity for each relation. Separate the
            # head/tail comparison because relation labels are assumed
            # one-hot encoded. The separation makes the following steps
            # easier.

            # All fields should match: (head/tail type, head/tail start, head/tail end).
            # entity_mask: [B, R, 6] -> [B, R]
            entity_mask = (y["entity"] == y_pred["entity"]).all(dim=-1)
            entity_mask = entity_mask.unsqueeze(dim=-1)

            # Use negative/sample mask to filter non-related relations.
            num_relation_types = len(self.relation_label_encoder)
            negative_mask = torch.arange(num_relation_types)[None, None, :]
            if index is None:
                negative_mask = negative_mask == self.negative_relation_index
            else:
                negative_mask = negative_mask != index
            y_sample_mask = y["mask"].unsqueeze(dim=-1).bool()
            y_pred_sample_mask = y_pred["mask"].unsqueeze(dim=-1).bool()
            y_mask = negative_mask | ~y_sample_mask
            y_pred_mask = negative_mask | ~y_pred_sample_mask
            y_label = y["label"].masked_fill(y_mask, 0)
            y_pred_label = y_pred["label"].masked_fill(y_pred_mask, 0)

            # A TP means:
            # 1. `y_pred_label` predicts every `y_label != 0` successfully.
            # 2. With matched head/tail entities.
            tp = y_pred_label.masked_fill((y_label == 0) | ~entity_mask, 0)
            fp = y_pred_label.masked_fill(tp > 0, 0)
            fn = y_label.masked_fill(tp > 0, 0)

            stat[0] += tp.sum().to(self._device)
            stat[1] += fp.sum().to(self._device)
            stat[2] += fn.sum().to(self._device)

        y_pred, y = output
        y_pred = {k: v.detach() for k, v in y_pred.items()}
        y = {k: v.detach() for k, v in y.items()}

        y = _expand_entities(y)
        y_pred = _expand_entities(y_pred)

        _update(y, y_pred, self.tpfpfn)
        for i in range(len(self.relation_label_encoder)):
            if i != self.negative_relation_index:
                _update(y, y_pred, self.typed_tpfpfn[i], i)

    @sync_all_reduce("tpfpfn:SUM", "typed_tpfpfn:SUM")
    def compute(self) -> dict[str, float]:
        metrics = prf1(self.tpfpfn[0], self.tpfpfn[1], self.tpfpfn[2])
        typed_metrics = {
            self.relation_label_encoder.decode_label(i): prf1(
                self.typed_tpfpfn[i][0],
                self.typed_tpfpfn[i][1],
                self.typed_tpfpfn[i][2],
            )
            for i in range(len(self.relation_label_encoder))
            if i != self.negative_relation_index
        }

        outputs = {"all": metrics, **typed_metrics}
        outputs = flatten_dict(outputs, ".")
        if self.prefix:
            outputs = {f"{self.prefix}.{k}": v for k, v in outputs.items()}

        return outputs
