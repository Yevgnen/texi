# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import BertTokenizerFast

from texi.preprocessing import LabelEncoder
from texi.pytorch.plm.spert import SpERT, SpERTDataset, SpERTParams, SpERTSampler
from texi.pytorch.plm.utils import plm_path
from texi.utils import ModeKeys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--params", type=SpERTParams.from_yaml, default="checkpoint/params.yaml"
    )
    parser.add_argument("--checkpoint", type=str, default="checkpoint/")
    parser.add_argument("--output", type=str, default="model.pt")

    return parser.parse_args()


def main(params, checkpoint, output):
    save_path = Path(checkpoint)

    tokenizer = BertTokenizerFast.from_pretrained(plm_path(params["pretrained_model"]))

    entity_label_encoder = LabelEncoder.load(save_path / "entity_labels.json")
    relation_label_encoder = LabelEncoder.load(save_path / "relation_labels.json")
    negative_entity_index = entity_label_encoder.encode_label(
        params["negative_entity_type"]
    )
    negative_relation_index = relation_label_encoder.encode_label(
        params["negative_relation_type"]
    )

    model = SpERT(
        params["pretrained_model"],
        params["embedding_dim"],
        len(entity_label_encoder),
        len(relation_label_encoder),
        negative_entity_index=negative_entity_index,
        dropout=params["dropout"],
        global_context_pooling=params["global_context_pooling"],
    )
    model.eval()

    negative_sampler = SpERTSampler(
        num_negative_entities=params["num_negative_entities"],
        num_negative_relations=params["num_negative_relations"],
        max_entity_length=params["max_entity_length"],
        negative_entity_type=params["negative_entity_type"],
        negative_relation_type=params["negative_relation_type"],
    )

    examples = [
        {
            "tokens": "Bill Gates was born in USA .".split(),
            "entities": [],
            "relations": [],
        }
    ]
    dataset = SpERTDataset(
        examples,
        negative_sampler,
        entity_label_encoder,
        relation_label_encoder,
        tokenizer,
        mode=ModeKeys.EVAL,
    )

    batch = dataset.collate_fn(examples)
    input_ = batch[1]

    traced_model = torch.jit.trace_module(
        model,
        {
            "infer": [
                input_["input_ids"],
                input_["attention_mask"],
                input_["token_type_ids"],
                input_["entity_mask"],
            ],
        },
        strict=False,
    )
    torch.jit.save(traced_model, output)


if __name__ == "__main__":
    args = parse_args()
    main(args.params, args.checkpoint, args.output)
