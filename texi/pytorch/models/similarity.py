# -*- coding: utf-8 -*-

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel

from texi.pytorch.attention import BidirectionalAttention
from texi.pytorch.losses import ManhattanSimilarity
from texi.pytorch.utils import length_to_mask


class SiameseLSTM(nn.Module):
    def __init__(
        self,
        embedded_size: int,
        hidden_size: int,
        num_layers: int = 2,
        batch_first: bool = True,
        dropout: float = 0.5,
        embedding: Optional[nn.Embedding] = None,
        vocab_size: Optional[int] = None,
        padding_idx: Optional[int] = 0,
    ):
        super().__init__()
        self.embedded_size = embedded_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.rnn_dropout = dropout if num_layers > 1 else 0
        if embedding is None:
            embedding = nn.Embedding(vocab_size, embedded_size, padding_idx=padding_idx)
        self.embedding = embedding
        self.vocab_size = vocab_size

        self.input_encoder = nn.LSTM(
            self.embedded_size,
            self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            bidirectional=True,
            dropout=self.rnn_dropout,
        )
        self.manhattan_similarity = ManhattanSimilarity()

    def _lstm_encode(self, lstm: nn.LSTM, inputs: torch.Tensor, lengths: torch.Tensor):
        packed = pack_padded_sequence(
            inputs, lengths, enforce_sorted=False, batch_first=self.batch_first
        )
        _, (hidden, _) = lstm(packed)
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_size)
        last_hidden = hidden[-1].transpose(0, 1)
        last_hidden = last_hidden.reshape(last_hidden.size()[0], -1)

        return last_hidden

    def _input_encoding(
        self,
        querys: torch.Tensor,
        docs: torch.Tensor,
        query_lengths: torch.Tensor,
        doc_lengths: torch.Tensor,
    ):
        return (
            self._lstm_encode(self.input_encoder, querys, query_lengths),
            self._lstm_encode(self.input_encoder, docs, doc_lengths),
        )

    def forward(self, inputs: Dict[str, torch.Tensor]):
        querys, docs = inputs["query"], inputs["doc"]
        query_lengths, doc_lengths = inputs["query_length"], inputs["doc_length"]

        # Token representation.
        query_embedded = self.embedding(querys)
        docembedded = self.embedding(docs)

        # Input encoding.
        query_encoded, doc_encoded = self._input_encoding(
            query_embedded, docembedded, query_lengths, doc_lengths
        )

        # Scoring.
        logits = self.manhattan_similarity(query_encoded, doc_encoded)

        return logits


class ESIM(nn.Module):
    def __init__(
        self,
        embedded_size: int,
        hidden_size: int,
        num_layers: int = 2,
        batch_first: bool = True,
        bidirectional: bool = True,
        dropout: float = 0.5,
        embedding: Optional[nn.Embedding] = None,
        vocab_size: Optional[int] = None,
        padding_idx: Optional[int] = 0,
    ):
        super().__init__()
        self.embedded_size = embedded_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.rnn_dropout = dropout if num_layers > 1 else 0
        if embedding is None:
            embedding = nn.Embedding(vocab_size, embedded_size, padding_idx=padding_idx)
        self.embedding = embedding
        self.vocab_size = vocab_size

        self.input_encoder = nn.LSTM(
            self.embedded_size,
            self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            bidirectional=self.bidirectional,
            dropout=self.rnn_dropout,
        )
        self.bidirectional_attention = BidirectionalAttention()
        self.composition_projection = nn.Sequential(
            nn.Linear(8 * self.hidden_size, self.hidden_size), nn.ReLU()
        )
        self.composition_encoder = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            bidirectional=self.bidirectional,
            dropout=self.rnn_dropout,
        )
        self.before_output = nn.Sequential(
            nn.Linear(8 * self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(self.hidden_size, 1)

    def _lstm_encode(self, lstm: nn.LSTM, inputs: torch.Tensor, lengths: torch.Tensor):
        packed = pack_padded_sequence(
            inputs, lengths, enforce_sorted=False, batch_first=self.batch_first
        )
        encoded, _ = lstm(packed)
        encoded, _ = pad_packed_sequence(encoded, batch_first=self.batch_first)

        return encoded

    def _input_encoding(
        self,
        querys: torch.Tensor,
        docs: torch.Tensor,
        query_lengths: torch.Tensor,
        doc_lengths: torch.Tensor,
    ):
        return (
            self._lstm_encode(self.input_encoder, querys, query_lengths),
            self._lstm_encode(self.input_encoder, docs, doc_lengths),
        )

    def _local_inference(
        self,
        query_encoded: torch.Tensor,
        doc_encoded: torch.Tensor,
        query_lengths: torch.Tensor,
        doc_lengths: torch.Tensor,
    ):
        query_semantics, doc_semantics = self.bidirectional_attention(
            query_encoded,
            ~length_to_mask(query_lengths, batch_first=self.batch_first),
            doc_encoded,
            ~length_to_mask(doc_lengths, batch_first=self.batch_first),
        )

        query_semantics = self._enhance_local_inference_features(
            query_encoded, query_semantics
        )
        doc_semantics = self._enhance_local_inference_features(
            doc_encoded, doc_semantics
        )

        return query_semantics, doc_semantics

    def _enhance_local_inference_features(
        self, encodded: torch.Tensor, local_relevance: torch.Tensor
    ):
        return torch.cat(
            [
                encodded,
                local_relevance,
                encodded - local_relevance,
                encodded * local_relevance,
            ],
            dim=-1,
        )

    def _inference_composition(
        self,
        query_information: torch.Tensor,
        doc_information: torch.Tensor,
        query_lengths: torch.Tensor,
        doc_lengths: torch.Tensor,
    ):

        return (
            self._lstm_encode(
                self.composition_encoder,
                self.composition_projection(query_information),
                query_lengths,
            ),
            self._lstm_encode(
                self.composition_encoder,
                self.composition_projection(doc_information),
                doc_lengths,
            ),
        )

    def _pooling(
        self,
        query_composited: torch.Tensor,
        doc_composited: torch.Tensor,
        query_lengths: torch.Tensor,
        doc_lengths: torch.Tensor,
    ):
        def _avg_pooling(tensor, mask):
            return torch.sum(tensor * mask, dim=int(self.batch_first))

        def _max_pooling(tensor, mask):
            return tensor.masked_fill(~mask, float("-inf")).max(
                dim=int(self.batch_first)
            )[0]

        query_mask = length_to_mask(
            query_lengths, batch_first=self.batch_first
        ).unsqueeze(dim=-1)
        doc_mask = length_to_mask(doc_lengths, batch_first=self.batch_first).unsqueeze(
            dim=-1
        )

        return torch.cat(
            [
                _avg_pooling(query_composited, query_mask),
                _max_pooling(query_composited, query_mask),
                _avg_pooling(doc_composited, doc_mask),
                _max_pooling(doc_composited, doc_mask),
            ],
            dim=-1,
        )

    def forward(self, inputs: Dict[str, torch.Tensor]):
        querys, docs = inputs["query"], inputs["doc"]
        query_lengths, doc_lengths = inputs["query_length"], inputs["doc_length"]

        # Token representation.
        query_embedded = self.embedding(querys)
        docembedded = self.embedding(docs)

        # Input encoding.
        query_encoded, doc_encoded = self._input_encoding(
            query_embedded, docembedded, query_lengths, doc_lengths
        )

        # Local inference.
        query_information, doc_information = self._local_inference(
            query_encoded, doc_encoded, query_lengths, doc_lengths
        )

        # Inference composition.
        query_composited, doc_composited = self._inference_composition(
            query_information, doc_information, query_lengths, doc_lengths
        )

        # Pooling.
        hidden = self._pooling(
            query_composited, doc_composited, query_lengths, doc_lengths
        )

        # Projection.
        before_output = self.before_output(hidden)
        logits = self.output(before_output)

        return logits


class BertSimilarity(nn.Module):
    def __init__(self, pretrained_model, pooling="mean_max", dropout=0.0):
        super().__init__()
        if isinstance(pretrained_model, str):
            self.bert = BertModel.from_pretrained(
                pretrained_model, num_labels=2, output_hidden_states=True
            )
        else:
            self.bert = pretrained_model
        self.pooling = pooling
        k = 2 if pooling == "mean_max" else 1
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(self.bert.config.hidden_size * k, 1)

    def forward(self, inputs):
        outputs = self.bert(**inputs)
        hidden = outputs[2][-2]
        mask = inputs["attention_mask"]

        hiddens = []
        if "mean" in self.pooling:
            mean_pooled = torch.sum(hidden * mask.unsqueeze(dim=-1), dim=1) / torch.sum(
                mask, dim=1, keepdim=True
            )
            hiddens += [mean_pooled]

        if "max" in self.pooling:
            max_pooled = torch.max(
                hidden + (1 - mask.unsqueeze(dim=-1)) * -1e10, dim=1
            )[0]
            hiddens += [max_pooled]

        hiddens = torch.cat(hiddens, dim=-1)
        hiddens = self.dropout(hiddens)
        logits = self.output(hiddens)

        return logits
