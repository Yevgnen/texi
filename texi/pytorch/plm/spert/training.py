# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import random
from typing import TYPE_CHECKING, Callable, Dict, Optional, Union

import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers import global_step_from_engine

from texi.apps.ner import SpERTVisualizer, entity_to_tuple, relation_to_tuple
from texi.preprocessing import LabelEncoder
from texi.pytorch.metrics import NerMetrics, ReMetrics
from texi.pytorch.plm.spert import predict
from texi.pytorch.training.params import Params
from texi.pytorch.training.trainer import Batch, MetricGroup, Trainer

try:
    import wandb
except ModuleNotFoundError:
    wandb = None


if TYPE_CHECKING:
    from ignite.contrib.handlers import WandBLogger
    from transformers import BertTokenizer, BertTokenizerFast


class SpERTParams(Params):
    # pylint: disable=too-many-instance-attributes
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pretrained_model = kwargs.get("pretrained_model", "bert-base-uncased")
        self.embedding_dim = kwargs.get("embedding_dim", 25)
        self.dropout = kwargs.get("dropout", 0.1)
        self.global_context_pooling = kwargs.get("global_context_pooling", "cls")
        self.negative_entity_type = kwargs.get("negative_entity_type", "NON_ENTITY")
        self.negative_relation_type = kwargs.get(
            "negative_relation_type", "NO_RELATION"
        )
        self.num_negative_entities = kwargs.get("num_negative_entities", 100)
        self.num_negative_relations = kwargs.get("num_negative_relations", 100)
        self.max_entity_length = kwargs.get("max_entity_length", 10)
        self.relation_filter_threshold = kwargs.get("relation_filter_threshold", 0.4)
        self.token_delimiter = kwargs.get("token_delimiter", " ")
        self.split_delimiter = kwargs.get("split_delimiter")

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


class SpERTEvalSampler(object):
    # pylint: disable=no-self-use
    def __init__(
        self,
        visualizer: SpERTVisualizer,
        tokenizer: Union[BertTokenizer, BertTokenizerFast],
        entity_label_encoder: LabelEncoder,
        negative_entity_index: int,
        relation_label_encoder: LabelEncoder,
        negative_relation_index: int,
        relation_filter_threshold: float,
        save_dir: str,
        sample_size: Optional[int] = None,
        wandb_logger: Optional[WandBLogger] = None,
    ):
        self.visualizer = visualizer
        self.tokenizer = tokenizer
        self.entity_label_encoder = entity_label_encoder
        self.negative_entity_index = negative_entity_index
        self.relation_label_encoder = relation_label_encoder
        self.negative_relation_index = negative_relation_index
        self.relation_filter_threshold = relation_filter_threshold
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.sample_size = sample_size
        self.wandb_logger = wandb_logger

        self.global_step_transform = None  # type: Callable
        self.reset()

    def reset(self):
        self.entity_samples = []
        self.relation_samples = []

    def started(self, _: Engine):
        self.reset()

    def _expand_entities(self, relations, entities):
        return [
            [
                {
                    "type": r["type"],
                    "head": sample_entities[r["head"]],
                    "tail": sample_entities[r["tail"]],
                }
                for r in sample_relations
            ]
            for sample_relations, sample_entities in zip(relations, entities)
        ]

    def _compare(self, y, y_pred, f, scores):
        y_trans = {*map(f, y)}
        y_pred_trans = {*map(f, y_pred)}

        items = []
        for yi_pred, yi_pred_tran, score in zip(y_pred, y_pred_trans, scores):
            if yi_pred_tran in y_trans:
                items += [(yi_pred, 0, score)]
            else:
                items += [(yi_pred, 1, -1)]

        for yi, yi_tran in zip(y, y_trans):
            if yi_tran not in y_pred_trans:
                items += [(yi, -1, -1)]

        return items

    def update(self, engine: Engine):
        target = engine.state.output["target"]
        input_ = engine.state.output["input"]
        output = engine.state.output["output"]

        (
            entity_predictions,
            entity_scores,
            relation_predictions,
            relation_scores,
        ) = predict(
            output["entity_logit"],
            input_["entity_sample_mask"],
            input_["entity_span"],
            self.entity_label_encoder,
            self.negative_entity_index,
            output["relation_logit"],
            output["relation"],
            output["relation_sample_mask"],
            self.relation_label_encoder,
            self.negative_relation_index,
            self.relation_filter_threshold,
            return_scores=True,
        )

        entity_targets, relation_targets = predict(
            target["entity_label"],
            target["entity_sample_mask"],
            target["entity_span"],
            self.entity_label_encoder,
            self.negative_entity_index,
            target["relation_label"],
            target["relation"],
            target["relation_sample_mask"],
            self.relation_label_encoder,
            self.negative_relation_index,
            self.relation_filter_threshold,
            return_scores=False,
        )

        relation_targets = self._expand_entities(relation_targets, entity_targets)
        relation_predictions = self._expand_entities(
            relation_predictions, entity_predictions
        )

        entity_samples, relation_samples = [], []
        for ts, e, e_pred, e_score, r, r_pred, r_score, in zip(
            input_["tokens"],
            entity_targets,
            entity_predictions,
            entity_scores,
            relation_targets,
            relation_predictions,
            relation_scores,
        ):
            entity_samples += [
                {
                    "tokens": ts[1:-1],
                    "entities": self._compare(e, e_pred, entity_to_tuple, e_score),
                }
            ]
            relation_samples += [
                {
                    "tokens": ts[1:-1],
                    "relations": self._compare(r, r_pred, relation_to_tuple, r_score),
                }
            ]

        self.entity_samples += entity_samples
        self.relation_samples += relation_samples

    def _sample(self, examples):
        if self.sample_size is not None:
            examples = random.sample(examples, min(len(examples), self.sample_size))

        return examples

    def export(self, _: Engine):
        epoch = self.global_step_transform(_, Events.EPOCH_COMPLETED)
        iteration = self.global_step_transform(_, Events.ITERATION_COMPLETED)

        entity_html = os.path.join(
            self.save_dir, f"entity_sample_epoch_{epoch}_iteration_{iteration}.html"
        )
        self.visualizer.export_entities(self._sample(self.entity_samples), entity_html)

        relation_html = os.path.join(
            self.save_dir, f"relation_sample_epoch_{epoch}_iteration_{iteration}.html"
        )
        self.visualizer.export_relations(
            self._sample(self.relation_samples), relation_html
        )

        if self.wandb_logger:
            if not wandb:
                raise RuntimeError("Install `wandb` package to enable HTML logging.")

            self.wandb_logger.log(
                {"Entity Extraction Examples": wandb.Html(open(entity_html))},
                step=iteration,
            )
            self.wandb_logger.log(
                {"Relation Extraction Examples": wandb.Html(open(relation_html))},
                step=iteration,
            )

    def setup(self, trainer: Engine, evaluator: Engine):
        self.global_step_transform = global_step_from_engine(trainer)

        evaluator.add_event_handler(Events.EPOCH_STARTED, self.reset)
        evaluator.add_event_handler(Events.ITERATION_COMPLETED, self.update)
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, self.export)
