# -*- coding: utf-8 -*-

import dataclasses
import json
import os
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union


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
