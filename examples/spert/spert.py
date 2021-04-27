# -*- coding: utf-8 -*-

import argparse
import logging
import os

from carton.logger import setup_logger
from carton.random import set_seed
from transformers import BertModel, BertTokenizerFast

from texi.apps.ner import SpERTVisualizer, encode_labels
from texi.datasets import JSONDatasets
from texi.pytorch.plm.spert import (
    SpERT,
    SpERTDataset,
    SpERTEvalSampler,
    SpERTLoss,
    SpERTParams,
    SpERTSampler,
    SpERTTrainer,
)
from texi.pytorch.plm.utils import get_pretrained_optimizer_and_scheduler

logger = logging.getLogger(__name__)


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
                dataset,
                tokenizer,
                entity_label_encoder,
                relation_label_encoder,
                params,
                mode == "train",
            )
            for mode, dataset in datasets.items()
        },
        train_batch_size=params["train_batch_size"],
        eval_batch_size=params["eval_batch_size"],
        num_workers=params["num_workers"],
    )

    return loaders


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--params", type=SpERTParams.from_yaml, default="spert.yaml")

    return parser.parse_args()  # pylint: disable=redefined-outer-name


def main():
    args = parse_args()
    params = args.params
    set_seed(params["seed"])

    os.makedirs(os.path.dirname(params["log_file"]), exist_ok=True)
    setup_logger(level=logging.INFO, filename=params["log_file"])

    datasets = JSONDatasets.from_dir(params.data_dir, array=True).load()

    tokenizer = BertTokenizerFast.from_pretrained(params["pretrained_model"])
    entity_label_encoder, relation_label_encoder = encode_labels(datasets.train)
    negative_entity_index = entity_label_encoder.add(params["negative_entity_type"])
    negative_relation_index = relation_label_encoder.add(
        params["negative_relation_type"]
    )

    loaders = get_dataloaders(
        datasets, tokenizer, entity_label_encoder, relation_label_encoder, params
    )

    bert = BertModel.from_pretrained(params["pretrained_model"])
    model = SpERT(
        bert,
        params["embedding_dim"],
        len(entity_label_encoder),
        len(relation_label_encoder),
        negative_entity_index=negative_entity_index,
        dropout=params["dropout"],
        global_context_pooling=params["global_context_pooling"],
    )
    model = model.to(params["device"])
    criteria = SpERTLoss()

    num_training_steps = (
        len(datasets.train) // params["train_batch_size"] * params["max_epochs"]
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
    eval_sampler = SpERTEvalSampler(
        SpERTVisualizer(),
        tokenizer,
        entity_label_encoder,
        negative_entity_index,
        relation_label_encoder,
        negative_relation_index,
        params["relation_filter_threshold"],
        params.sample_dir,
    )
    eval_sampler.setup(trainer.trainer, trainer.evaluators["val_evaluator"])
    trainer.run()


if __name__ == "__main__":
    main()
