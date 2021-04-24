# -*- coding: utf-8 -*-

from typing import Dict, Iterable, List, Mapping, Sequence, Union


def split_example(
    example: Mapping, delimiters: Union[str, Iterable[str]]
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
                current_entities += [entities[entity_index]]
                entity_index += 1

            if entity_index < len(entities) and entities[entity_index]["start"] <= i:
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
                    raise RuntimeError(
                        "Relation must not across delimiters,"
                        f" delimiters: {delimiters}, relation: {relation}"
                    )

                if in_range == 0:
                    current_relations += [relation]
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
