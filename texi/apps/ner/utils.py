# -*- coding: utf-8 -*-

import copy
import dataclasses
import json
import os
import re
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from carton.collections import dict_to_tuple

from texi.preprocessing import LabelEncoder


def entity_to_tuple(entity: Mapping) -> Tuple:
    return dict_to_tuple(entity, ["type", "start", "end"])


def relation_to_tuple(relation: Mapping, entities: Optional[Mapping] = None) -> Tuple:
    def _convert_entity(entity):
        if isinstance(entity, Mapping):
            return entity_to_tuple(entity)

        if entities:
            return entities[entity]

        return entity

    return (
        relation["type"],
        _convert_entity(relation["head"]),
        _convert_entity(relation["tail"]),
    )


def expand_entities(
    relations: Iterable[Mapping], entities: Sequence[Mapping]
) -> List[Dict]:
    expanded = [
        {
            "type": relation["type"],
            "head": dict(entities[relation["head"]]),
            "tail": dict(entities[relation["tail"]]),
        }
        for relation in relations
    ]
    return expanded


def collapse_entities(
    relations: Iterable[Mapping], entities: Iterable[Mapping]
) -> List[Mapping]:
    entity_indices = {entity_to_tuple(entity): i for i, entity in enumerate(entities)}
    collapsed = [
        {
            "type": relation["type"],
            "head": entity_indices[entity_to_tuple(relation["head"])],
            "tail": entity_indices[entity_to_tuple(relation["tail"])],
        }
        for relation in relations
    ]

    return collapsed


def encode_labels(examples: Iterable[Mapping]) -> Tuple[LabelEncoder, LabelEncoder]:
    entity_label_encoder = LabelEncoder()
    relation_label_encoder = LabelEncoder()

    for example in examples:
        for entity in example["entities"]:
            entity_label_encoder.add(entity["type"])

        for relation in example["relations"]:
            relation_label_encoder.add(relation["type"])

    return entity_label_encoder, relation_label_encoder


def convert_pybrat_example(example: Mapping) -> Dict:
    # NOTE:
    # 1. ID fields are kept.
    # 2. Entities are sort before conversion.

    # Convert tokens.
    tokens = list(example["text"])

    # Record entity indices.
    entities = sorted(example["entities"], key=lambda x: x["start"])

    # Convert relation.
    entity_indices = {x["id"]: i for i, x in enumerate(entities)}
    relation = [
        {
            "id": x["id"],
            "type": x["type"],
            "head": entity_indices[x["arg1"]["id"]],
            "tail": entity_indices[x["arg2"]["id"]],
        }
        for x in example["relations"]
    ]

    # Convert entities.
    entities = [
        {
            "id": x["id"],
            "type": x["type"],
            "start": x["start"],
            "end": x["end"],
        }
        for x in entities
    ]

    example = {
        "id": example["id"],
        "tokens": tokens,
        "entities": entities,
        "relations": relation,
    }

    return example


def load_pybrat_examples(dirname: str, *args, **kwargs) -> List[Dict]:
    # pylint: disable=import-outside-toplevel
    from pybrat.parser import BratParser

    parser = BratParser(*args, **kwargs)
    parsed = parser.parse(dirname)

    examples = []
    for parsed_example in parsed:
        example = dataclasses.asdict(parsed_example)
        example = convert_pybrat_example(example)
        examples += [example]

    return examples


def convert_pybrat_examples(
    input_dir: str,
    output_dir: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    shuffle: bool = True,
    random_state: Optional[int] = None,
    **kwargs,
) -> None:
    # pylint: disable=import-outside-toplevel
    from sklearn.model_selection import train_test_split

    def _optional_split(data, size):
        if size > 0:
            first, second = train_test_split(
                data, test_size=size, shuffle=shuffle, random_state=random_state
            )
        else:
            first, second = data, None

        return first, second

    # Load examples.
    examples = load_pybrat_examples(input_dir, **kwargs)

    # Split examples.
    train, test = _optional_split(examples, test_size)
    train, val = _optional_split(train, val_size)

    # Dump examples.
    prefix = os.path.basename(input_dir)
    os.makedirs(output_dir, exist_ok=True)
    datasets = {
        "train": train,
        "val": val,
        "test": test,
    }
    for mode, dataset in datasets.items():
        if dataset is not None:
            with open(os.path.join(output_dir, f"{prefix}_{mode}.json"), mode="w") as f:
                json.dump(dataset, f, ensure_ascii=False)


def check_example(example: Mapping) -> bool:
    if len(example["tokens"]) < 1:
        raise ValueError("`example` has no tokens")

    num_tokens = len(example["tokens"])
    for entity in example["entities"]:
        if entity["start"] >= entity["end"]:
            raise ValueError(f"Invalid entity span: {entity}")

        if (
            entity["start"] >= num_tokens
            or entity["end"] >= num_tokens
            or entity["start"] < 0
            or entity["end"] < 0
        ):
            raise ValueError(f"Entity token out of bound: {entity}")

    num_entities = len(example["entities"])
    for relation in example["relations"]:
        if relation["head"] >= num_entities or relation["tail"] >= num_entities:
            raise ValueError(f"Entity not found for relation: {relation}")

        if relation["head"] == relation["tail"]:
            raise ValueError(
                f"Relation should have different head and tail: {relation}"
            )

    return True


def filter_example_tokens(
    example: Mapping, filters: Iterable[Union[str, re.Pattern, Callable[[str], bool]]]
) -> Dict:
    if not hasattr(filters, "__iter__") or isinstance(filters, str):
        filters = [filters]

    for f in filters:
        if not isinstance(f, (str, re.Pattern)) and not callable(f):
            raise ValueError(
                "Filter should be str, re.Pattern or Callable,"
                f" not: {f.__class__.__name__}"
            )

    def _filter(x):
        for f in filters:
            if isinstance(f, re.Pattern):
                if re.match(f, x):
                    return True

            elif isinstance(f, str):
                if f == x:
                    return True

            elif callable(f):
                if f(x):
                    return True

        return False

    backup = example
    example = dict(copy.deepcopy(example))
    entities = sorted(enumerate(example["entities"]), key=lambda x: x[1]["start"])

    entity_index = 0
    num_entities = len(entities)
    tokens = example["tokens"]
    num_tokens = len(tokens)
    i = 0
    while i < num_tokens:
        if _filter(tokens[i]):
            while entity_index < num_entities and entities[entity_index][1]["end"] <= i:
                entity_index += 1

            if entity_index < num_entities and entities[entity_index][1]["start"] <= i:
                entity_tokens = tokens[
                    entities[entity_index][1]["start"] : entities[entity_index][1][
                        "end"
                    ]
                ]
                raise RuntimeError(f"Can not filter entity tokens: {entity_tokens}")

            j = entity_index
            while j < num_entities:
                entities[j][1]["start"] -= 1
                entities[j][1]["end"] -= 1
                j += 1

            tokens.pop(i)
            num_tokens -= 1
        else:
            i += 1

    example["tokens"] = tokens
    if entities:
        entities = list(list(zip(*sorted(entities, key=lambda x: x[0])))[1])
    example["entities"] = entities

    assert len(example["entities"]) == len(backup["entities"]), "Mismatched entity list"
    for i, entity in enumerate(example["entities"]):
        assert (
            example["tokens"][entity["start"] : entity["end"]]
            == backup["tokens"][
                backup["entities"][i]["start"] : backup["entities"][i]["end"]
            ]
        ), "Mismatched entity spans"

    return example


def split_example(
    example: Mapping, delimiters: Union[str, Iterable[str]], ignore_errors: bool = False
) -> List[Dict]:
    if isinstance(delimiters, str):
        delimiters = {delimiters}
    else:
        delimiters = set(delimiters)

    # Sorting entities may change indices.
    if not example["tokens"]:
        raise ValueError("`example` should at least contain one token")

    entities = list(example["entities"])
    relations = sorted(example["relations"], key=lambda x: (x["head"], x["tail"]))

    splits = []
    current_tokens, current_entities, current_relations = [], [], []
    entity_index, relation_index = 0, 0
    for i, token in enumerate(example["tokens"] + [next(iter(delimiters))]):
        current_tokens += [token]

        if token in delimiters:
            # Collect entities.
            entity_indices = {}  # type: Dict[int, int]
            while entity_index < len(entities) and entities[entity_index]["end"] <= i:
                entity_indices[entity_index] = len(entity_indices)
                entity = entities[entity_index]
                entity_start = len(current_tokens) - (i - entity["start"]) - 1
                entity_end = len(current_tokens) - (i - entity["end"]) - 1
                current_entity = {
                    "type": entities[entity_index]["type"],
                    "start": entity_start,
                    "end": entity_end,
                }
                current_entities += [current_entity]
                entity_index += 1
                entity_start += 1

            if entity_index < len(entities) and entities[entity_index]["start"] <= i:
                if ignore_errors:
                    entity_index += 1
                else:
                    raise RuntimeError(
                        "Entity must not contains delimiters,"
                        f" delimiters: {delimiters}, entity: {entities[entity_index]}"
                    )

            # Collect relations.
            while relation_index < len(relations):
                relation = relations[relation_index]
                head_index = entity_indices.get(relation["head"])
                tail_index = entity_indices.get(relation["tail"])
                in_range = bool(head_index is None) + bool(tail_index is None)
                if in_range == 1:
                    if ignore_errors:
                        relation_index += 1
                    else:
                        raise RuntimeError(
                            "Relation must not across delimiters,"
                            f" delimiters: {delimiters}, relation: {relation}"
                        )

                if in_range == 0:
                    current_relation = {
                        "type": relation["type"],
                        "head": head_index,
                        "tail": tail_index,
                    }
                    current_relations += [current_relation]
                    relation_index += 1
                else:
                    # This also implies that invalid relations will be drop.
                    break

            # Create new split.
            split = {
                "tokens": current_tokens,
                "entities": current_entities,
                "relations": current_relations,
            }
            splits += [split]

            # Reset states.
            current_tokens, current_entities, current_relations = [], [], []

    if len(splits[-1]["tokens"]) == 1:
        splits.pop()
    else:
        splits[-1]["tokens"].pop()

    return splits


def merge_examples(examples: Sequence[Mapping]) -> Dict[str, List]:
    if len(examples) < 1:
        raise ValueError("At least one example must be given to merge")

    tokens = []  # type: List[Dict]
    entities = []  # type: List[Dict]
    relations = []  # type: List[Dict]

    for example in examples:
        token_offset = len(tokens)

        # Collect tokens.
        tokens += example["tokens"]

        # Collect entities.
        entity_indices = {}
        num_entities_so_far = len(entities)
        for i, entity in enumerate(example["entities"]):
            new_entity = {
                "type": entity["type"],
                "start": entity["start"] + token_offset,
                "end": entity["end"] + token_offset,
            }
            entity_indices[i] = i + num_entities_so_far
            entities += [new_entity]

        # Collect relations.
        for relation in example["relations"]:
            new_relation = {
                "type": relation["type"],
                # `dict.get` is not used implies invalid relations should fail.
                "head": entity_indices[relation["head"]],
                "tail": entity_indices[relation["tail"]],
            }
            relations += [new_relation]

    return {
        "tokens": tokens,
        "entities": entities,
        "relations": relations,
    }


def texify_example(example: Dict, delimiter: str) -> Dict:
    entities = example["entities"]
    if not entities:
        return {
            "tokens": delimiter.join(example["tokens"]),
            "entities": entities,
            "relations": example["relations"],
        }

    num_tokens = len(example["tokens"])
    delimiter_length = len(delimiter)
    entity_index = 0
    entity = entities[entity_index]
    new_tokens, new_entities = [], []
    start = -1
    char_offset = 0
    for i, token in enumerate(example["tokens"]):
        if i == entity["end"]:
            if start < 0:
                raise ValueError(f"Invalid entity: {entity}")

            new_enitty = {
                "type": entity["type"],
                "start": start,
                "end": char_offset - delimiter_length,
            }
            new_entities += [new_enitty]

            entity_index += 1
            if entity_index < len(entities):
                start = -1
                entity = entities[entity_index]

        if i == entity["start"]:
            start = char_offset

        new_tokens += [token]
        char_offset += len(token)

        if i < num_tokens - 1:
            new_tokens += [delimiter]
            char_offset += delimiter_length

    return {
        "tokens": "".join(new_tokens),
        "entities": new_entities,
        "relations": example["relations"],
    }
