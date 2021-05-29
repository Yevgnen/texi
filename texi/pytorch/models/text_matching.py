# -*- coding: utf-8 -*-

from typing import Union

import torch
import torch.nn as nn
from transformers import BertModel

from texi.pytorch.plm.pooling import get_pooling
from texi.pytorch.utils import plm_path


class SBertBiEncoder(nn.Module):
    def __init__(self, bert: Union[BertModel, str], pooling: str = "mean") -> None:
        super().__init__()
        if isinstance(bert, str):
            bert = BertModel.from_pretrained(plm_path(bert), add_pooling_layer=False)
        self.bert = bert

        self.pooling = get_pooling(pooling)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        token_type_ids: torch.LongTensor,
    ) -> torch.Tensor:
        n, batch_size, max_length = input_ids.size()

        def _flatten(t):
            return t.view(-1, max_length)

        # input_ids, attention_mask, token_type_ids: [N, B, L] -> [NB, L]
        input_ids = _flatten(input_ids)
        attention_mask = _flatten(attention_mask)
        token_type_ids = _flatten(token_type_ids)

        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = self.pooling(bert_output, attention_mask)
        pooled = pooled.view(n, batch_size, -1)

        return pooled


class SBertForClassification(nn.Module):
    def __init__(
        self,
        bert: Union[BertModel, str],
        pooling: str = "mean",
        num_labels: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.sbert_encoder = SBertBiEncoder(bert, pooling=pooling)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            3 * self.sbert_encoder.bert.config.hidden_size,
            1 if num_labels == 2 else num_labels,
        )
        self.num_labels = num_labels

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        token_type_ids: torch.LongTensor,
    ) -> torch.Tensor:
        def _check_size(t):
            if t.ndim != 3 or t.size()[0] != 2:
                raise ValueError(
                    "`input_ids`, `attention_mask`, `token_type_ids` should all"
                    " have size: [2, batch_size, max_length]"
                )

        _check_size(input_ids)
        _check_size(attention_mask)
        _check_size(token_type_ids)

        hidden = self.sbert_encoder(input_ids, attention_mask, token_type_ids)

        u, v = hidden
        feature = torch.cat([u, v, torch.abs(u - v)], dim=-1)

        logit = self.classifier(feature)
        if self.num_labels == 2:
            logit = logit.squeeze(dim=-1)

        return logit


class SBertForRegression(nn.Module):
    def __init__(
        self,
        bert: Union[BertModel, str],
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.sbert_encoder = SBertBiEncoder(bert, pooling=pooling)
        self.cosine_similarity = nn.CosineSimilarity()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        token_type_ids: torch.LongTensor,
    ) -> torch.Tensor:
        def _check_size(t):
            if t.ndim != 3 or t.size()[0] != 2:
                raise ValueError(
                    "Input tensor should have size: [2, batch_size, max_length]"
                )

        _check_size(input_ids)
        _check_size(attention_mask)
        _check_size(token_type_ids)

        hidden = self.sbert_encoder(input_ids, attention_mask, token_type_ids)

        u, v = hidden
        score = self.cosine_similarity(u, v)

        return score
