# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import BertModel


class BertForQuestionAnswering(nn.Module):
    def __init__(self, pretrained_model: str, **kwargs):
        super().__init__()
        if isinstance(pretrained_model, str):
            self.bert = BertModel.from_pretrained(pretrained_model, **kwargs)
        else:
            self.bert = pretrained_model
        self.projection = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        hiddens, *_ = self.bert(**inputs)
        logits = self.projection(hiddens)

        return logits
