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

    def _match_spans(self, span_index, last_hidden_state):
        batch_size = span_index.size(dim=0)
        index = torch.arange(batch_size, device=span_index.device)[:, None, None]
        span_input = last_hidden_state[index, span_index].flatten(2)
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
        span_logit = self._match_spans(span_index, last_hidden_state)

        start_logit = start_logit.squeeze(dim=-1)
        end_logit = end_logit.squeeze(dim=-1)
        span_logit = span_logit.squeeze(dim=-1)

        return start_logit, end_logit, span_logit
