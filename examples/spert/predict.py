# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import functools
import json
import os
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Union, cast

import ignite.distributed as idist
from ignite.engine.engine import Engine
from ignite.engine.events import Events
from torch import nn
from torchlight.dataset import get_dataloader
from torchlight.preprocessing import LabelEncoder
from torchlight.training import create_evaluator, run
from torchlight.utils.file import load_checkpoint, plm_path
from torchlight.utils.phase import ModeKeys
from transformers import BertTokenizerFast

from texi.apps.ner import split_example
from texi.apps.ner.utils import merge_examples
from texi.datasets.dataset import Dataset
from texi.pytorch.models.spert import SpERT, SpERTParams
from texi.pytorch.models.spert.dataset import SpERTCollator, SpERTDataset
from texi.pytorch.models.spert.prediction import predict as predict_relations
from texi.pytorch.models.spert.sampler import SpERTSampler
from texi.pytorch.models.spert.training import eval_step


def add_dummy_labels(x: MutableMapping) -> dict:
    x["entities"] = []
    x["relations"] = []

    return dict(x)


def merge_tokens_with_predictions(dataset: Dataset, predictions: Sequence[Mapping]):
    for example, prediction in zip(dataset, predictions):
        example["entities"] = prediction[0]
        example["relations"] = prediction[1]


def save_predictions(
    predictions: Sequence[Mapping], output: Union[str, bytes, os.PathLike]
) -> None:
    with open(output, mode="w") as f:
        json.dump(predictions, f, ensure_ascii=False)


def create_predictor(
    model: nn.Module,
    params: SpERTParams,
    entity_label_encoder: LabelEncoder,
    negative_entity_index: int,
    relation_label_encoder: LabelEncoder,
    negative_relation_index: int,
) -> Engine:
    def predict_step(engine, model, batch):
        output = eval_step(engine, model, batch)

        input_ = output["input"]
        output = output["output"]

        (entity_predictions, relation_predictions) = predict_relations(
            output["entity_logit"],
            input_["entity_sample_mask"],
            input_["entity_span"],
            entity_label_encoder,
            negative_entity_index,
            output["relation_logit"],
            output["relation"],
            output["relation_sample_mask"],
            relation_label_encoder,
            negative_relation_index,
            params["relation_filter_threshold"],
        )

        engine.state.predictions += list(zip(entity_predictions, relation_predictions))

    engine = create_evaluator(predict_step, params, model, "test")

    @engine.on(Events.STARTED)
    def init_states(engine) -> None:
        engine.state.predictions = []

    return engine


def predict(
    local_rank: int,
    params: SpERTParams,
    test_file: str,
    output: str,
    save_path: Path,
    checkpoint: Path,
) -> None:
    # Load datasets.
    dataset = SpERTDataset.from_json(test_file, array=True).load()
    dataset.map(add_dummy_labels)
    dataset.eval()

    if params.split_delimiter:
        dataset.split(
            functools.partial(
                split_example, delimiters=params.split_delimiter, ignore_errors=True
            )
        )
    if params.max_length > 0:
        dataset.mask(lambda x: len(x["tokens"]) < params["max_length"])

    # Get text/label encoders.
    tokenizer = BertTokenizerFast.from_pretrained(plm_path(params["pretrained_model"]))
    entity_label_encoder = LabelEncoder.load(save_path / "entity_labels.json")
    relation_label_encoder = LabelEncoder.load(save_path / "relation_labels.json")
    negative_entity_index = cast(
        int, entity_label_encoder.encode_label(params["negative_entity_type"])
    )
    negative_relation_index = cast(
        int, relation_label_encoder.encode_label(params["negative_relation_type"])
    )

    # Get data dataflow.
    negative_sampler = SpERTSampler(
        max_entity_length=params["max_entity_length"],
        num_negative_entities=params["num_negative_entities"],
        num_negative_relations=params["num_negative_relations"],
        negative_entity_type=params["negative_entity_type"],
        negative_relation_type=params["negative_relation_type"],
    )
    dataflow = get_dataloader(
        dataset,
        collate_fn=SpERTCollator(
            negative_sampler,
            tokenizer,
            entity_label_encoder,
            relation_label_encoder,
            mode=ModeKeys.PREDICT,
        ),
        batch_size=params["predict_batch_size"],
        num_workers=params["num_workers"],
        pin_memory=True,
    )

    # Create model.
    model = SpERT(
        params["pretrained_model"],
        params["embedding_dim"],
        len(entity_label_encoder),
        len(relation_label_encoder),
        negative_entity_index=negative_entity_index,
        dropout=params["dropout"],
        global_context_pooling=params["global_context_pooling"],
    ).to(idist.device())
    load_checkpoint(model, checkpoint)

    # Run predict.
    engine = create_predictor(
        model,
        params,
        entity_label_encoder,
        negative_entity_index,
        relation_label_encoder,
        negative_relation_index,
    )
    engine.run(dataflow)

    # Merge tokens with predicted entities and relations.
    merge_tokens_with_predictions(dataset, engine.state.predictions)

    if params.max_length > 0:
        dataset.unmask()

    if params.split_delimiter:
        dataset.merge(merge_examples)

    # Save predictions.
    save_predictions(dataset.examples, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save_path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)

    # pylint: disable=redefined-outer-name
    args = parser.parse_args()
    args.params = SpERTParams.from_yaml(args.save_path / "params.yaml")
    args.params.log_file = None
    args.checkpoint = args.save_path / args.checkpoint

    return args


if __name__ == "__main__":
    args = parse_args()

    run(predict, args.params, args.input, args.output, args.save_path, args.checkpoint)
