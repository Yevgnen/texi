# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn
from transformers.utils.dummy_pt_objects import BertModel

from texi.pytorch.plm.utils import plm_path


class Mrc4Ner(nn.Module):
    def __init__(self, bert: Union[str, BertModel]):
        super().__init__()
        if isinstance(bert, str):
            bert = BertModel.from_pretrained(plm_path(bert), add_pooling_layer=False)
        self.bert = bert

        self.start_classifier = nn.Linear(self.bert.config.hidden_size, 1)
        self.end_classifier = nn.Linear(self.bert.config.hidden_size, 1)
        self.span_classifier = nn.Linear(2 * self.bert.config.hidden_size, 1)

    def _match_spans(self, start_logit, end_logit, span_index, last_hidden_state):
        # start, end: [B, L, H]
        start = torch.softmax(start_logit, dim=-1).round()
        end = torch.softmax(end_logit, dim=-1).round()

        max_length = start_logit.size(dim=1)
        index = torch.arange(max_length, device=start_logit.device)
        start = last_hidden_state[index[:, None], span_index[:, 0][None, :]]
        end = last_hidden_state[index[:, None], span_index[:, 1][None, :]]

        # span_logit: [B, S, 2]
        span_input = torch.cat([start, end], dim=-1)
        span_logit = self.span_classifier(span_input)

        return span_logit

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        token_type_ids: torch.LongTensor,
        span_index: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        last_hidden_state = bert_output["last_hidden_state"]

        start_logit = self.start_classifier(last_hidden_state)
        end_logit = self.end_classifier(last_hidden_state)

        span_logit = self._match_spans(
            start_logit, end_logit, span_index, last_hidden_state
        )

        return start_logit, end_logit, span_logit
