# -*- coding: utf-8 -*-

from typing import Dict

import torch.nn as nn

from texi.preprocessing import LabelEncoder
from texi.pytorch.plm.spert.prediction import decode_entities, decode_relations, predict
from texi.pytorch.training.trainer import Batch, MetricGroup, Trainer
from texi.pytorch.training.params import Params


class SpERTParams(Params):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pretrained_model = kwargs.pop("pretrained_model", "bert-base-uncased")
        self.embedding_dim = kwargs.pop("embedding_dim", 25)
        self.dropout = kwargs.pop("dropout", 0.1)
        self.negative_entity_type = kwargs.pop("negative_entity_type", "NON_ENTITY")
        self.negative_relation_type = kwargs.pop(
            "negative_relation_type", "NO_RELATION"
        )
        self.num_negative_entities = kwargs.pop("num_negative_entities", 100)
        self.num_negative_relations = kwargs.pop("num_negative_relations", 100)
        self.max_entity_length = kwargs.pop("max_entity_length", 10)
        self.relation_filter_threshold = kwargs.pop("relation_filter_threshold", 0.4)

    def __getitem__(self, key):
        return self.__dict__[key]


class SpERTTrainer(Trainer):
    def __init__(
        self,
        entity_label_encoder: LabelEncoder,
        negative_entity_index: int,
        relation_label_encoder: LabelEncoder,
        negative_relation_index: int,
        relation_filter_threshold: float,
    ):
        super().__init__()
        self.entity_label_encoder = entity_label_encoder
        self.negative_entity_index = negative_entity_index
        self.relation_label_encoder = relation_label_encoder
        self.negative_relation_index = negative_relation_index
        self.relation_filter_threshold = relation_filter_threshold

    def get_metrics(self, train: bool = True) -> MetricGroup:
        return {}

    def train_step(
        self, net: nn.Module, batch: Batch, loss_function: nn.Module
    ) -> Dict:
        output = net(
            batch["input_ids"],
            batch["attention_mask"],
            batch["token_type_ids"],
            batch["entity_mask"],
            batch["relation"],
            batch["relation_context_mask"],
            batch["relation_sample_mask"],
        )

        loss = loss_function(
            output["entity_logit"],
            batch["entity_label"],
            batch["entity_sample_mask"],
            output["relation_logit"],
            batch["relation_label"],
            batch["relation_sample_mask"],
        )

        return {"batch": batch, "loss": loss}

    def eval_step(self, net: nn.Module, batch: Batch) -> Dict:
        output = net.infer(
            batch["input_ids"],
            batch["attention_mask"],
            batch["token_type_ids"],
            batch["entity_mask"],
        )

        entity_predictions, relation_predictions = list(
            zip(
                *predict(
                    output["entity_logit"],
                    batch["entity_mask"],
                    batch["entity_token_span"],
                    self.entity_label_encoder,
                    self.negative_entity_index,
                    output["relation_logit"],
                    output["relation"],
                    output["relation_sample_mask"],
                    self.relation_label_encoder,
                    self.negative_relation_index,
                    self.relation_filter_threshold,
                )
            )
        )

        entity_sample_mask = batch["entity_mask"].sum(dim=-1) > 0
        entity_targets = decode_entities(
            batch["entity_label"],
            entity_sample_mask,
            batch["entity_token_span"],
            self.entity_label_encoder,
            self.negative_entity_index,
        )

        relation_targets = decode_relations(
            batch["relation_label"],
            batch["relation"],
            batch["relation_sample_mask"],
            self.relation_label_encoder,
            self.negative_relation_index,
            self.relation_filter_threshold,
        )

        return (
            entity_targets,
            entity_predictions,
            relation_targets,
            relation_predictions,
        )
