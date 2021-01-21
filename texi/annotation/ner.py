# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
from typing import Any, Dict, Tuple, Union

import numpy as np

from texi.annotation.parser import DualFileAnnotationParser
from texi.tagger import IOB2

_Entities = Union[Dict[str, Any], Tuple[str, str, int, int]]


class BratParser(DualFileAnnotationParser):
    def __init__(self, **kwargs):
        kwargs["text_ext"] = kwargs.get("text_ext", ".txt")
        kwargs["annotation_ext"] = kwargs.get("annotation_ext", ".ann")
        super().__init__(**kwargs)

    def _parse_ann(self, ann, filters):
        with open(ann) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line[0] in filters:
                    splits = line.split()
                    if len(splits) == filters[line[0]]:
                        yield splits

    def parse_text(self, filename):
        with open(filename) as f:
            return f.read().rstrip()

    def parse_entities(self, filename):
        return self._parse_ann(filename, {"T": 5})

    def parse_relations(self, ann):
        return self._parse_ann(ann, {"R": 4})

    # TODO: 2020-03-12 Move this method to DualFileAnnotationParser.
    def parse_re(self, dirname, seps="。", ignores="\n 　", tagger=None):
        if tagger is None:
            tagger = IOB2()

        seps = set(seps)
        ignores = set(ignores)

        for ann, txt in self._iter_dir_samples(dirname):
            # Parse texts.
            text = self.parse_text(txt)

            # Parse entities and relations.
            entities, relations = [], []
            for splits in self._parse_ann(ann, {"T": 5, "R": 4}):
                if len(splits) == 5:
                    id_, type_, start, end, word = splits
                    entities += [(id_, type_, word, float(start), float(end))]
                elif len(splits) == 4:
                    _, relation, head, tail = splits
                    relations += [
                        (
                            self._relations.get(relation, relation),
                            head.split(":")[1],
                            tail.split(":")[1],
                        )
                    ]

            # Refine entity indices.
            entity_indices = np.asarray([x[-2:] for x in entities], dtype=np.int64)
            chars = []
            for i, char in reversed(list(enumerate(text))):
                if char in ignores:
                    entity_indices[entity_indices > i] -= 1
                else:
                    chars += [char]
            chars = list(reversed(chars))
            entities = [
                (*entity[:-2], *indices)
                for indices, entity in zip(entity_indices, entities)
            ]

            # Iterate all split examples.
            example_chars = []
            offset = 0
            last_i = -1
            for i, char in enumerate(chars):
                example_chars += [char]

                if (char in seps or i == len(text) - 1) and example_chars:
                    example_entities = {
                        x[0]: tuple(list(x[1:-2]) + [x[-2] - offset, x[-1] - offset])
                        for x in entities
                        if x[-2] > last_i and x[-1] <= i
                    }
                    example_tags = tagger.encode(
                        example_chars, example_entities.values()
                    )[1]

                    example_relations = []
                    for relation, head, tail in relations:
                        head = example_entities.get(head)
                        tail = example_entities.get(tail)
                        if not head or not tail:
                            continue

                        example_relations += [(head, relation, tail)]

                    yield example_chars, example_tags, example_relations

                    offset += len(example_chars)
                    example_chars = []
                    last_i = i


class XmlParser(DualFileAnnotationParser):
    def __init__(self, **kwargs):
        self.text_parser = kwargs.get("text_parser")
        kwargs["text_ext"] = kwargs.get("text_ext", ".xml")
        kwargs["annotation_ext"] = kwargs.get("annotation_ext", ".ent")
        self.text_offset = kwargs.pop("text_offset", 0)
        super().__init__(**kwargs)

    def parse_text(self, filename):
        if callable(self.text_parser):
            with open(filename, mode="r") as f:
                return self.text_parser(f.read())

        return ET.parse(filename).getroot().text

    def parse_entities(self, filename):
        with open(filename, mode="r") as f:
            id_ = -1
            for line in f:
                line = line.strip()
                if not line:
                    continue

                word_str, pos_str, type_str = line.split()[:3]
                word = word_str.split("=")[1]
                type_ = type_str.split("=")[1]
                start, end = [int(x) for x in pos_str.split("=")[1].split(":")]
                start -= self.text_offset
                end -= self.text_offset
                id_ += 1
                yield (id_, type_, start, end, word)
