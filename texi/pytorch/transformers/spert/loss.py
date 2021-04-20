# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class SpERTLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.entity_loss = nn.CrossEntropyLoss(reduction="none")
        self.relation_loss = nn.BCEWithLogitsLoss(reduction="none")

    def _entity_loss(self, entity_logits, entity_labels, entity_sample_masks):
        entity_logits = entity_logits.transpose(-1, -2)
        entity_loss = self.entity_loss(entity_logits, entity_labels)
        entity_loss.masked_fill_(entity_sample_masks == 0, 0)
        entity_loss = entity_loss.sum() / entity_sample_masks.sum()

        return entity_loss

    def _relation_loss(self, relation_logits, relation_labels, relation_sample_masks):
        if relation_logits.size(1) == 0:
            return relation_logits.new_zeros(1)

        relation_loss = self.relation_loss(relation_logits, relation_labels.float())
        relation_loss.masked_fill_(relation_sample_masks.unsqueeze(-1) == 0, 0)
        relation_loss = relation_loss.sum() / relation_sample_masks.sum()

        return relation_loss

    def forward(
        self,
        entity_logits: torch.FloatTensor,
        entity_labels: torch.LongTensor,
        entity_sample_masks: torch.LongTensor,
        relation_logits: torch.FloatTensor,
        relation_labels: torch.LongTensor,
        relation_sample_masks: torch.LongTensor,
    ) -> torch.FloatTensor:
        entity_loss = self._entity_loss(
            entity_logits, entity_labels, entity_sample_masks
        )

        relation_loss = self._relation_loss(
            relation_logits, relation_labels, relation_sample_masks
        )

        loss = entity_loss + relation_loss

        return loss
