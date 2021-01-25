# -*- coding: utf-8 -*-

import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel

from texi.pytorch.rnn import LSTM


class BiLstmCrf(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_labels,
        embedded_size=100,
        hidden_size=100,
        num_layers=2,
        char_embedding=None,
        dropout=0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_labels = num_labels
        if char_embedding is None:
            char_embedding = nn.Embedding(vocab_size, embedded_size)
        self.char_embedding = char_embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.encoder = LSTM(
            embedded_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
        )
        self.fc = nn.Linear(2 * hidden_size, num_labels)

    def forward(self, inputs):
        x = inputs["token"]
        lengths = inputs["length"]

        embedded = self.char_embedding(x)
        embedded = self.embedding_dropout(embedded)
        hidden, _ = self.encoder(embedded, lengths)
        logits = self.fc(hidden)

        return logits


class BertForSequenceLabeling(nn.Module):
    def __init__(self, pretrained_model, **kwargs):
        super().__init__()
        num_labels = kwargs.get("num_labels")
        if num_labels is None:
            raise KeyError("Required key `num_labels` not give.")

        if isinstance(pretrained_model, str):
            self.bert = BertModel.from_pretrained(
                pretrained_model, num_labels=num_labels
            )
        else:
            self.bert = pretrained_model

        self.output = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, inputs):
        inputs = {
            key: value
            for key, value in inputs.items()
            if key not in {"offset_mapping", "tag_mask"}
        }

        outputs = self.bert(**inputs)
        logits = self.output(outputs[0])

        return logits


class SequenceCrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, input_, target, tag_mask):
        assert input_.size()[:-1] == tag_mask.size()
        assert target.size() == tag_mask.size()

        tag_mask = tag_mask.view(-1).bool()
        input_ = input_.view(-1, input_.size()[-1])[tag_mask]
        target = target.view(-1)[tag_mask]

        return super().forward(input_, target)


class CRFForPreTraining(CRF):
    def __init__(self, num_tags, batch_first=True):
        super().__init__(num_tags, batch_first=batch_first)

    def forward(self, emissions, tags, mask, reduction="sum"):
        emissions = emissions[:, 1:, :]
        tags = tags[:, 1:]
        mask = mask[:, 1:].bool()

        return -super().forward(emissions, tags, mask, reduction=reduction)

    def decode(self, emissions, mask):
        return super().decode(emissions, mask.bool())
