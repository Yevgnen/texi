# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class Mrc4NerLoss(nn.Module):
    def __init__(
        self, alpha: float = 0.3, beta: float = 0.3, gamma: float = 0.4
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.index_loss = nn.BCEWithLogitsLoss()
        self.span_loss = nn.BCEWithLogitsLoss(reduction="none")

    def _index_loss(self, logit, target):
        return self.index_loss(logit, target.float())

    def _span_loss(self, span_logit, span_label, span_mask):
        loss = self.span_loss(span_logit, span_label.float())
        loss.masked_fill_(~span_mask.bool(), 0)

        return loss.mean()

    def forward(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        span_label: torch.Tensor,
        span_mask: torch.LongTensor,
        start_logit: torch.Tensor,
        end_logit: torch.Tensor,
        span_logit: torch.Tensor,
    ) -> torch.Tensor:
        start_loss = self._index_loss(start_logit, start)
        end_loss = self._index_loss(end_logit, end)
        span_loss = self._span_loss(span_logit, span_label, span_mask)

        loss = self.alpha * start_loss + self.beta * end_loss + self.gamma * span_loss

        return loss
