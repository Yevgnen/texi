# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn
from transformers import BertModel

from texi.pytorch.plm.pooling import get_pooling
from texi.pytorch.plm.utils import plm_path


class BertForSequenceClassification(nn.Module):
    def __init__(
        self,
        bert: Union[BertModel, str],
        dropout: float = 0.1,
        pooling: str = "cls",
        num_labels: int = 2,
    ):
        super().__init__()
        if isinstance(bert, str):
            bert = BertModel.from_pretrained(plm_path(bert), add_pooling_layer=False)
        self.bert = bert

        self.pooling = get_pooling(pooling)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            bert.config.hidden_size, 1 if num_labels == 2 else num_labels
        )

        self.num_labels = num_labels

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        token_type_ids: torch.LongTensor,
    ) -> torch.Tensor:
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        pooled = self.pooling(bert_output, attention_mask)
        drop = self.dropout(pooled)
        logit = self.classifier(drop)

        if self.num_labels == 2:
            logit = logit.squeeze(dim=-1)

        return logit
