# -*- coding: utf-8 -*-

from typing import Mapping, Optional, Sequence

from carton.palette import Colors
from carton.random import random_colors


def spacy_visual_ner(
    examples: Sequence[Mapping],
    filename: Optional[str] = None,
    colors: Optional[Mapping[str, str]] = None,
    token_sep: str = " ",
):
    # FIXME: Does not work for token inputs.

    # pylint: disable=import-outside-toplevel
    from spacy import displacy

    # Generate spacy inputs.
    spacy_data = [
        {
            "text": token_sep.join(example["tokens"]),
            "ents": [
                {"start": x["start"], "end": x["end"], "label": x["type"]}
                for x in example["entities"]
            ],
            "title": example.get("id", f"#{i}"),
        }
        for i, example in enumerate(examples)
    ]

    # Give each type a color.
    if not colors:
        types = set(x["type"] for example in examples for x in example["entities"])
        colors = random_colors(Colors.PRESETS, num=len(types))
        colors = dict(zip(types, colors))

    display_fn = displacy.render if filename else displacy.serve
    rendered = display_fn(
        spacy_data, style="ent", manual=True, options={"colors": colors}, page=True
    )

    if filename:
        with open(filename, mode="w") as f:
            f.writelines(rendered)
