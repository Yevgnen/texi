# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class TransformerLayerDistillationLoss(nn.Module):
    def __init__(
        self,
        selected_layers: Optional[list[int]] = None,
        weights: Optional[list[float]] = None,
        loss: nn.Module = nn.MSELoss(reduction="none"),
    ) -> None:
        super().__init__()
        self.selected_layers = selected_layers
        if weights:
            self.weights = torch.tensor(weights, dtype=torch.int64)
        else:
            self.weights = None  # type: ignore
        self.loss = loss

    def _select_layer(self, tensors):
        if self.selected_layers is None:
            return tensors

        return tuple(tensors[layer] for layer in self.selected_layers)

    def forward(
        self,
        teacher_hidden_states: tuple[torch.Tensor],
        teacher_attentions: tuple[torch.Tensor],
        student_hidden_states: tuple[torch.Tensor],
        student_attentions: tuple[torch.Tensor],
    ) -> torch.Tensor:
        teacher_hidden_states = self._select_layer(teacher_hidden_states)
        teacher_attentions = self._select_layer(teacher_attentions)

        teacher_hidden_state = torch.stack(teacher_hidden_states)
        student_hidden_state = torch.stack(student_hidden_states)
        hidden_state_loss = self.loss(student_hidden_state, teacher_hidden_state)

        teacher_attention = torch.stack(teacher_attentions)
        student_attention = torch.stack(student_attentions)
        attention_loss = self.loss(student_attention, teacher_attention)

        if self.weights is not None:
            hidden_state_loss *= self.weights.view(-1, 1, 1, 1)
            attention_loss *= self.weights.view(-1, 1, 1, 1, 1)

        num_heads = student_attention.size(dim=2)
        hidden_state_loss = hidden_state_loss.mean()
        attention_loss = attention_loss.mean() / num_heads

        loss = hidden_state_loss + attention_loss

        return loss


class EmbeddingLayerDistillationLoss(nn.Module):
    def __init__(self, loss: nn.Module = nn.MSELoss(reduction="mean")) -> None:
        super().__init__()
        self.loss = loss

    def forward(
        self,
        teacher_embedding_output: tuple[torch.Tensor],
        student_embedding_output: tuple[torch.Tensor],
    ) -> torch.Tensor:
        loss = self.loss(teacher_embedding_output, student_embedding_output)

        return loss
