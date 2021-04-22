# -*- coding: utf-8 -*-

import argparse
import logging
import os

from carton.logger import setup_logger
from carton.params import Params
from carton.random import set_seed
from transformers import AdamW, BertModel, BertTokenizerFast

from texi.preprocessing import LabelEncoder
from texi.pytorch.plm.spert import (
    SpERT,
    SpERTDataset,
    SpERTLoss,
    SpERTParams,
    SpERTSampler,
    SpERTTrainer,
)

logger = logging.getLogger(__name__)


def read_dataset(path):
    import json

    with open(path) as f:
        examples = json.load(f)
    return examples


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


def get_dataloaders(
    datasets, tokenizer, entity_label_encoder, relation_label_encoder, params
):
    negative_sampler = SpERTSampler(
        num_negative_entities=params["num_negative_entities"],
        num_negative_relations=params["num_negative_relations"],
        max_entity_length=params["max_entity_length"],
        negative_entity_type=params["negative_entity_type"],
        negative_relation_type=params["negative_relation_type"],
    )
    train_dataset = SpERTDataset(
        datasets["train"],
        negative_sampler,
        entity_label_encoder,
        relation_label_encoder,
        tokenizer,
        train=True,
    )
    val_dataset = SpERTDataset(
        datasets["val"],
        negative_sampler,
        train_dataset.entity_label_encoder,
        train_dataset.relation_label_encoder,
        tokenizer,
    )
    test_dataset = SpERTDataset(
        datasets["test"],
        negative_sampler,
        train_dataset.entity_label_encoder,
        train_dataset.relation_label_encoder,
        tokenizer,
    )

    loaders = SpERTDataset.get_dataloaders(
        {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        },
        train_batch_size=params["train_batch_size"],
        eval_batch_size=params["eval_batch_size"],
    )

    return loaders


def get_model():
    return


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

    # parser = BratParser(error="ignore")
    # examples = parser.parse("../../../projects/mc/data/all/")
    # examples = list(map(dataclasses.asdict, examples))
    # for example in examples:
    #     example["text"] = list(example["text"])
    #
    # train_val, test = train_test_split(examples, test_size=params["test_size"])
    # train, val = train_test_split(train_val, test_size=params["val_size"])
    #
    # # train = read_dataset("../../spert/data/datasets/conll04/conll04_train.json")
    # # val = read_dataset("../../spert/data/datasets/conll04/conll04_dev.json")
    # # test = val

    train = [
        {
            "tokens": ["BillGates", "was", "born", "in", "America", "."],
            "entities": [
                {"type": "per", "start": 0, "end": 1},
                {"type": "prep", "start": 3, "end": 4},
                {"type": "loc", "start": 4, "end": 5},
            ],
            "relations": [{"type": "born in", "head": 0, "tail": 2}],
        },
        {
            "tokens": ["John", "loves", "Mary", "."],
            "entities": [
                {"type": "per", "start": 0, "end": 1},
                {"type": "per", "start": 2, "end": 3},
            ],
            "relations": [{"type": "loves", "head": 0, "tail": 1}],
        },
        {
            "tokens": ["Stop", "talking", "and", "get", "out", "here"],
            "entities": [],
            "relations": [],
        },
    ]
    val = train
    test = train

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
    optimizer = AdamW(model.parameters(), lr=params["lr"])

    trainer = SpERTTrainer(
        entity_label_encoder,
        negative_entity_index,
        relation_label_encoder,
        negative_relation_index,
        params["relation_filter_threshold"],
    )
    trainer.setup(params, loaders, model, criteria, optimizer)
    trainer.run()


if __name__ == "__main__":
    main()
