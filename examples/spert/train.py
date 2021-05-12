# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import functools
import os
from collections.abc import Mapping
from typing import Optional, Union

import ignite.distributed as idist
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from transformers import AdamW, BertTokenizer, BertTokenizerFast

from texi.apps.ner import SpERTVisualizer, encode_labels, split_example
from texi.apps.ner.span_utils import RelationFilter
from texi.datasets import JSONDatasets
from texi.datasets.dataset import Dataset, Datasets
from texi.preprocessing import LabelEncoder
from texi.pytorch.plm.spert import (
    SpERT,
    SpERTDataset,
    SpERTEnv,
    SpERTEvalSampler,
    SpERTLoss,
    SpERTParams,
    SpERTSampler,
)
from texi.pytorch.plm.spert.training import eval_step, train_step
from texi.pytorch.plm.utils import get_pretrained_optimizer_and_scheduler, plm_path
from texi.pytorch.training.training import (
    create_engines,
    describe_dataflows,
    run,
    setup_env,
)
from texi.utils import ModeKeys


def get_dataset(
    examples: Dataset,
    tokenizer: Union[BertTokenizer, BertTokenizerFast],
    entity_label_encoder: LabelEncoder,
    relation_label_encoder: LabelEncoder,
    params: SpERTParams,
    mode: ModeKeys,
    relation_argument_types: Optional[Mapping] = None,
) -> SpERTDataset:
    negative_sampler = SpERTSampler(
        num_negative_entities=params["num_negative_entities"],
        num_negative_relations=params["num_negative_relations"],
        max_entity_length=params["max_entity_length"],
        negative_entity_type=params["negative_entity_type"],
        negative_relation_type=params["negative_relation_type"],
        relation_argument_types=relation_argument_types,
    )
    dataset = SpERTDataset(
        examples,
        negative_sampler,
        entity_label_encoder,
        relation_label_encoder,
        tokenizer,
        mode=mode,
        device=idist.device(),
    )

    return dataset


def get_dataflows(
    datasets: Datasets,
    tokenizer: Union[BertTokenizerFast, BertTokenizer],
    entity_label_encoder: LabelEncoder,
    relation_label_encoder: LabelEncoder,
    params: SpERTParams,
    relation_argument_types: Optional[Mapping] = None,
) -> dict[str, DataLoader]:
    # `pin_memory = False` is required since `auto_dataloader` set
    # `pin_memory` to True by default, but we have moved tensors to GPU
    # by passing `device` to Dataset.
    dataflows = SpERTDataset.get_dataloaders(
        {
            mode: get_dataset(
                dataset,
                tokenizer,
                entity_label_encoder,
                relation_label_encoder,
                params,
                ModeKeys.TRAIN if mode == "train" else ModeKeys.EVAL,
                relation_argument_types=relation_argument_types,
            )
            for mode, dataset in datasets.items()
            if dataset is not None
        },
        train_batch_size=params["train_batch_size"],
        eval_batch_size=params["eval_batch_size"],
        num_workers=params["num_workers"],
        sort_key=lambda x: len(x["tokens"]),
        pin_memory=False,
    )

    return dataflows


def initialize(
    params: SpERTParams,
    num_entity_types: int,
    num_relation_types: int,
    negative_entity_index: int,
    num_train_examples: int,
    relation_argument_types: Optional[Mapping] = None,
) -> tuple[SpERT, SpERTLoss, AdamW, LambdaLR]:
    model = SpERT(
        params["pretrained_model"],
        params["embedding_dim"],
        num_entity_types,
        num_relation_types,
        negative_entity_index=negative_entity_index,
        dropout=params["dropout"],
        global_context_pooling=params["global_context_pooling"],
        relation_argument_types=relation_argument_types,
    )

    num_training_steps = (
        num_train_examples // params["train_batch_size"] * params["max_epochs"]
    )
    warmup_steps = params["lr_warmup"] * num_training_steps
    optimizer, lr_scheduler = get_pretrained_optimizer_and_scheduler(
        model, params["lr"], params["weight_decay"], warmup_steps, num_training_steps
    )
    criteria = SpERTLoss()

    model = idist.auto_model(model)
    optimizer = idist.auto_optim(optimizer)
    criteria = criteria.to(idist.device())

    return model, criteria, optimizer, lr_scheduler


def training(local_rank: int, params: SpERTParams) -> None:
    if idist.get_rank() == 0:
        setup_env(params)

    # Load datasets.
    datasets = JSONDatasets.from_dir(params.data_dir, array=True).load()
    if params.split_delimiter:
        datasets.split(
            functools.partial(
                split_example, delimiters=params.split_delimiter, ignore_errors=True
            )
        )

    if params.max_length > 0:
        datasets.mask(lambda x: len(x["tokens"]) < params["max_length"])

    # Get text/label encoders.
    tokenizer = BertTokenizerFast.from_pretrained(plm_path(params["pretrained_model"]))
    entity_label_encoder, relation_label_encoder = encode_labels(datasets.train)
    negative_entity_index = entity_label_encoder.add(params["negative_entity_type"])
    negative_relation_index = relation_label_encoder.add(
        params["negative_relation_type"]
    )
    entity_label_encoder.save(os.path.join(params.save_path, "entity_labels.json"))
    relation_label_encoder.save(os.path.join(params.save_path, "relation_labels.json"))

    relation_filter = None
    if params["relation_argument_types"]:
        relation_filter = RelationFilter(
            params["relation_argument_types"],
            entity_label_encoder,
            relation_label_encoder,
        )

    # Get data dataflows.
    dataflows = get_dataflows(
        datasets,
        tokenizer,
        entity_label_encoder,
        relation_label_encoder,
        params,
        relation_filter.types if relation_filter else None,
    )
    describe_dataflows(dataflows)

    # Create model.
    model, criteria, optimizer, lr_scheduler = initialize(
        params,
        len(entity_label_encoder),
        len(relation_label_encoder),
        negative_entity_index,
        len(datasets.train),
        relation_filter.type_ids if relation_filter else None,
    )

    # Prepare trainer.
    env = SpERTEnv(
        entity_label_encoder,
        negative_entity_index,
        relation_label_encoder,
        negative_relation_index,
        params["relation_filter_threshold"],
    )

    trainer, evaluators, loggers = create_engines(
        params,
        train_step,
        eval_step,
        dataflows,
        model,
        criteria,
        optimizer,
        lr_scheduler,
        train_metrics=env.get_metrics(train=True),
        eval_metrics=env.get_metrics(train=False),
        with_handlers=True,
    )

    # Setup evaluation sampler.
    if idist.get_rank() == 0:
        eval_sampler = SpERTEvalSampler(
            SpERTVisualizer(params["token_delimiter"]),
            tokenizer,
            entity_label_encoder,
            negative_entity_index,
            relation_label_encoder,
            negative_relation_index,
            params["relation_filter_threshold"],
            params.sample_dir,
            wandb_logger=loggers.get("wandb_logger"),
        )
        eval_sampler.setup(trainer, evaluators["val"])

    # Train!
    trainer.run(dataflows["train"], max_epochs=params["max_epochs"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--params", type=SpERTParams.from_yaml, default="spert.yaml")

    return parser.parse_args()


if __name__ == "__main__":
    run(training, parse_args().params)
