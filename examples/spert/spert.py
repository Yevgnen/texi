# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os

from carton.logger import setup_logger
from carton.params import Params
from carton.random import set_seed
from transformers import BertModel, BertTokenizerFast

from texi.preprocessing import LabelEncoder
from texi.pytorch.plm.spert import (
    SpERT,
    SpERTDataset,
    SpERTLoss,
    SpERTParams,
    SpERTSampler,
    SpERTTrainer,
)
from texi.pytorch.plm.utils import get_pretrained_optimizer_and_scheduler

logger = logging.getLogger(__name__)


def read_dataset(path):
    with open(path) as f:
        return json.load(f)


def get_label_encoders(train, negative_entity_type, negative_relation_type):
    entity_label_encoder = LabelEncoder(
        [e["type"] for x in train for e in x["entities"]]
    )
    entity_label_encoder.add(negative_entity_type)
    relation_label_encoder = LabelEncoder(
        [r["type"] for x in train for r in x["relations"]]
    )
    relation_label_encoder.add(negative_relation_type)

    return entity_label_encoder, relation_label_encoder


def get_dataset(
    examples, tokenizer, entity_label_encoder, relation_label_encoder, params, train
):
    negative_sampler = SpERTSampler(
        num_negative_entities=params["num_negative_entities"],
        num_negative_relations=params["num_negative_relations"],
        max_entity_length=params["max_entity_length"],
        negative_entity_type=params["negative_entity_type"],
        negative_relation_type=params["negative_relation_type"],
        train=train,
    )
    dataset = SpERTDataset(
        examples,
        negative_sampler,
        entity_label_encoder,
        relation_label_encoder,
        tokenizer,
        train=train,
    )

    return dataset


def get_dataloaders(
    datasets, tokenizer, entity_label_encoder, relation_label_encoder, params
):
    loaders = SpERTDataset.get_dataloaders(
        {
            mode: get_dataset(
                datasets[mode],
                tokenizer,
                entity_label_encoder,
                relation_label_encoder,
                params,
                mode == "train",
            )
            for mode in datasets
        },
        train_batch_size=params["train_batch_size"],
        eval_batch_size=params["eval_batch_size"],
    )

    return loaders


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--params", type=Params.from_yaml, default="spert.yaml")

    return parser.parse_args()  # pylint: disable=redefined-outer-name


def main():
    args = parse_args()
    params = SpERTParams(**args.params)
    set_seed(params["seed"])

    os.makedirs(os.path.dirname(params.log_file), exist_ok=True)
    setup_logger(level=logging.INFO, filename=params.log_file)

    train = read_dataset(
        "../../../repos/spert/data/datasets/conll04/conll04_train.json"
    )
    val = read_dataset("../../../repos/spert/data/datasets/conll04/conll04_dev.json")
    test = read_dataset("../../../repos/spert/data/datasets/conll04/conll04_test.json")

    logger.info("Train size: %d", len(train))
    logger.info("Val size: %d", len(val))
    logger.info("Test size: %d", len(test))

    datasets = {"train": train, "val": val, "test": test}

    tokenizer = BertTokenizerFast.from_pretrained(params["pretrained_model"])
    entity_label_encoder, relation_label_encoder = get_label_encoders(
        train, params["negative_entity_type"], params["negative_relation_type"]
    )

    loaders = get_dataloaders(
        datasets, tokenizer, entity_label_encoder, relation_label_encoder, params
    )

    negative_entity_index = entity_label_encoder.encode_label(
        params["negative_entity_type"]
    )
    negative_relation_index = relation_label_encoder.encode_label(
        params["negative_relation_type"]
    )

    bert = BertModel.from_pretrained(params["pretrained_model"])
    model = SpERT(
        bert,
        params["embedding_dim"],
        len(entity_label_encoder),
        len(relation_label_encoder),
        negative_entity_index=negative_entity_index,
        dropout=params["dropout"],
    )
    model = model.to(params["device"])
    criteria = SpERTLoss()

    num_training_steps = (
        len(loaders["train"].dataset)
        // params["train_batch_size"]
        * params["max_epochs"]
    )
    warmup_steps = params["lr_warmup"] * num_training_steps
    optimizer, lr_scheduler = get_pretrained_optimizer_and_scheduler(
        model, params["lr"], params["weight_decay"], warmup_steps, num_training_steps
    )

    trainer = SpERTTrainer(
        entity_label_encoder,
        negative_entity_index,
        relation_label_encoder,
        negative_relation_index,
        params["relation_filter_threshold"],
    )
    trainer.setup(
        params, loaders, model, criteria, optimizer, lr_scheduler=lr_scheduler
    )
    trainer.run()


if __name__ == "__main__":
    main()
