# -*- coding: utf-8 -*-

from typing import Dict, Iterable, List, Mapping, Union


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
