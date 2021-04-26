# -*- coding: utf-8 -*-

from typing import Dict

import torch.nn as nn

from texi.preprocessing import LabelEncoder
from texi.pytorch.metrics import NerMetrics, ReMetrics
from texi.pytorch.training.params import Params
from texi.pytorch.training.trainer import Batch, MetricGroup, Trainer


class SpERTParams(Params):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pretrained_model = kwargs.pop("pretrained_model", "bert-base-uncased")
        self.embedding_dim = kwargs.pop("embedding_dim", 25)
        self.dropout = kwargs.pop("dropout", 0.1)
        self.global_context_pooling = kwargs.get("global_context_pooling", "cls")
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
        if train:
            return {}

        return {
            "ner": NerMetrics(
                self.entity_label_encoder,
                self.negative_entity_index,
                prefix="NER",
                output_transform=lambda outputs: {
                    "y": {
                        "label": outputs["target"]["entity_label"],
                        "span": outputs["target"]["entity_span"],
                        "mask": outputs["target"]["entity_sample_mask"],
                    },
                    "y_pred": {
                        "label": outputs["output"]["entity_logit"].argmax(dim=-1),
                        "span": outputs["input"]["entity_span"],
                        "mask": outputs["input"]["entity_sample_mask"],
                    },
                },
            ),
            "re": ReMetrics(
                self.relation_label_encoder,
                self.negative_relation_index,
                self.relation_filter_threshold,
                prefix="RE",
                output_transform=lambda outputs: {
                    "y": {
                        "label": outputs["target"]["relation_label"],
                        "pair": outputs["target"]["relation"],
                        "mask": outputs["target"]["relation_sample_mask"],
                        "entity_span": outputs["target"]["entity_span"],
                        "entity_label": outputs["target"]["entity_label"],
                    },
                    "y_pred": {
                        "label": outputs["output"]["relation_logit"],
                        "pair": outputs["output"]["relation"],
                        "mask": outputs["output"]["relation_sample_mask"],
                        "entity_span": outputs["input"]["entity_span"],
                        "entity_label": outputs["output"]["entity_logit"].argmax(
                            dim=-1
                        ),
                    },
                },
            ),
        }

    def train_step(
        self, _: Engine, net: nn.Module, batch: Batch, loss_function: nn.Module
    ) -> Dict:
        output = net(
            batch["input_ids"],
            batch["attention_mask"],
            batch["token_type_ids"],
            batch["entity_mask"],
            batch["relation"],
            batch["relation_context_mask"],
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

    def eval_step(self, _: Engine, net: nn.Module, batch: Batch) -> Dict:
        target, input_ = batch
        output = net.infer(
            input_["input_ids"],
            input_["attention_mask"],
            input_["token_type_ids"],
            input_["entity_mask"],
        )

        return {
            "target": target,
            "input": input_,
            "output": output,
        }
