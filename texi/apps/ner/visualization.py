# -*- coding: utf-8 -*-

import itertools
from typing import Dict, Mapping, Optional, Sequence

from carton.palette import Colors
from carton.random import random_colors

from texi.tagger import Tagger


def _spacy_visual_ner(examples, filename, colors):
    # pylint: disable=import-outside-toplevel
    from spacy import displacy

    # Generate spacy inputs.
    spacy_data = [
        {
            "tokens": example["tokens"],
            "ents": [
                {"start": x["start"], "end": x["end"], "label": x["tag"]}
                for x in example["chunks"]
            ],
            "title": example.get("id", f"#{i}"),
        }
        for i, example in enumerate(examples)
    ]

    # Give each type a color.
    if not colors:
        types = set(x["tag"] for example in examples for x in example["chunks"])
        colors = random_colors(Colors.PRESETS, num=len(types))
        colors = dict(zip(types, colors))

    display_fn = displacy.render if filename else displacy.serve
    rendered = display_fn(
        spacy_data, style="ent", manual=True, options={"colors": colors}, page=True
    )

    if filename:
        with open(filename, mode="w") as f:
            f.writelines(rendered)


def visualize_ner(
    examples: Sequence[Mapping],
    filename: Optional[str] = None,
    colors: Optional[Mapping[str, str]] = None,
    sort: bool = False,
    drop_duplicates: bool = False,
) -> None:
    if drop_duplicates:
        tagger = Tagger("iob2")
        examples = [
            dict(x)
            for x in {
                (("tokens", x["tokens"]), ("tag", tuple(x["tag"])))
                for x in map(tagger.encode, examples)
            }
        ]
        examples = [*map(tagger.decode, examples)]

    if sort:
        examples = sorted(examples, key=lambda x: len(x["tokens"]))

    _spacy_visual_ner(examples, filename, colors)


def visualize_ner_prediction(
    y: Sequence[Mapping],
    y_pred: Sequence[Mapping],
    filename: Optional[str] = None,
    colors: Optional[Mapping[str, str]] = None,
    ignore_corrections: bool = False,
) -> None:
    if ignore_corrections:
        y, y_pred = zip(*[(yi, yi_pred) for yi, yi_pred in zip(y, y_pred)])

    data = itertools.chain.from_iterable(zip(y, y_pred))

    _spacy_visual_ner(data, filename, colors)
