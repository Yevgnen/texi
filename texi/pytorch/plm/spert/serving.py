# -*- coding: utf-8 -*-

from pathlib import Path

from transformers import BertTokenizerFast
from ts.torch_handler.base_handler import BaseHandler

from texi.preprocessing import LabelEncoder
from texi.pytorch.plm.spert import SpERTDataset, SpERTSampler
from texi.pytorch.plm.spert.prediction import predict as predict_relations
from texi.pytorch.plm.spert.training import SpERTParams
from texi.pytorch.plm.utils import plm_path
from texi.utils import ModeKeys


class SpERTHandler(BaseHandler):
    def initialize(self, context):
        super().initialize(context)

        properties = context.system_properties
        model_dir = Path(properties.get("model_dir"))

        self.params = SpERTParams.from_yaml(model_dir / "params.yaml")

        self.tokenizer = BertTokenizerFast.from_pretrained(model_dir)
        self.entity_label_encoder = LabelEncoder.load(model_dir / "entity_labels.json")
        self.relation_label_encoder = LabelEncoder.load(
            model_dir / "relation_labels.json"
        )
        self.negative_entity_index = self.entity_label_encoder.encode_label(
            self.params["negative_entity_type"]
        )
        self.negative_relation_index = self.relation_label_encoder.encode_label(
            self.params["negative_relation_type"]
        )

        self.negative_sampler = SpERTSampler(
            num_negative_entities=self.params["num_negative_entities"],
            num_negative_relations=self.params["num_negative_relations"],
            max_entity_length=self.params["max_entity_length"],
            negative_entity_type=self.params["negative_entity_type"],
            negative_relation_type=self.params["negative_relation_type"],
        )

    def preprocess(self, data):
        examples = [
            {
                "tokens": x["data"].decode().split(),
                "entities": [],
                "relations": [],
            }
            for x in data
        ]

        dataset = SpERTDataset(
            examples,
            self.negative_sampler,
            self.entity_label_encoder,
            self.relation_label_encoder,
            self.tokenizer,
            mode=ModeKeys.EVAL,
            device=self.device,
        )

        return dataset.collate_fn(dataset.examples)

    def inference(self, data, *args, **kwargs):
        input_ = data[1]

        output = self.model.infer(
            input_["input_ids"],
            input_["attention_mask"],
            input_["token_type_ids"],
            input_["entity_mask"],
        )

        (entity_predictions, relation_predictions) = predict_relations(
            output["entity_logit"],
            data["entity_sample_mask"],
            data["entity_span"],
            self.entity_label_encoder,
            self.negative_entity_index,
            output["relation_logit"],
            output["relation"],
            output["relation_sample_mask"],
            self.relation_label_encoder,
            self.negative_relation_index,
            self.params["relation_filter_threshold"],
        )

        return list(zip(entity_predictions, relation_predictions))
