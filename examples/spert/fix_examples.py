# -*- coding: utf-8 -*-

import json
import os
import re

from texi.apps.ner import filter_example_tokens
from texi.datasets import JSONDatasets

input_dir = "hopi"
output_dir = "hopi_fixed"

datasets = JSONDatasets.from_dir(input_dir, array=True).load()


def write_examples(examples, filename):
    with open(filename, mode="w") as f:
        json.dump(examples, f, ensure_ascii=False)


os.makedirs(output_dir, exist_ok=True)
for mode, dataset in datasets.items():
    examples = []
    for example in dataset:
        example = filter_example_tokens(example, filters=re.compile(r"^\s*$"))
        examples += [example]
    write_examples(examples, os.path.join(output_dir, f"{mode}.json"))
