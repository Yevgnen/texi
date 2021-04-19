# -*- coding: utf-8 -*-

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from pybrat.parser import Example


@dataclasses.dataclass
class Token(object):
    start: int
    end: int
    phrase: str
    index: Optional[int] = None
    id: Optional[str] = None

    def as_dict(self):
        return dataclasses.asdict(self)

    def __getitem__(self, key):
        return self.phrase[key]

    def __iter__(self):
        yield from iter(self.phrase)

    def __len__(self):
        return len(self.phrase)


@dataclasses.dataclass
class TokenSpan(object):
    tokens: List[Token]

    @property
    def start(self):
        return self.tokens[0].start

    @property
    def end(self):
        return self.tokens[-1].end

    @property
    def span(self):
        return (self.start, self.end)

    def as_dict(self):
        return list(map(dataclasses.asdict, self.tokens))

    def __getitem__(self, key):
        return self.tokens[key]

    def __iter__(self):
        yield from iter(self.tokens)

    def __len__(self):
        return len(self.tokens)


@dataclasses.dataclass
class EntityType(object):
    name: str

    def __str__(self):
        return self.name


@dataclasses.dataclass
class Entity(object):
    type: EntityType
    token_span: TokenSpan
    phrase: str
    index: Optional[int] = None
    id: Optional[str] = None

    @property
    def start(self):
        return self.token_span.start

    @property
    def end(self):
        return self.token_span.end

    @property
    def span(self):
        return self.token_span.span

    def as_dict(self):
        return {
            "type": str(self.type),
            "token_span": self.token_span.as_dict(),
            "phrase": self.phrase,
            "index": self.index,
            "id": self.id,
        }


@dataclasses.dataclass
class RelationType(object):
    name: str

    def __str__(self):
        return self.name


@dataclasses.dataclass
class Relation(object):
    type: RelationType
    head: Entity
    tail: Entity
    index: Optional[int] = None
    id: Optional[str] = None

    def as_dict(self):
        return {
            "type": str(self.type),
            "head": self.head.as_dict(),
            "tail": self.tail.as_dict(),
            "id": self.id,
        }


@dataclasses.dataclass
class Document(object):
    tokens: List[Token]
    entities: List[Entity] = dataclasses.field(default_factory=list)
    relations: List[Relation] = dataclasses.field(default_factory=list)
    id: Optional[str] = None
    doc: Optional[str] = None

    def as_dict(self):
        return {
            "doc": self.doc,
            "tokens": list(map(dataclasses.asdict, self.tokens)),
            "entities": [x.as_dict() for x in self.entities],
            "relations": [x.as_dict() for x in self.relations],
            "id": self.id,
        }

    def __len__(self):
        return len(self.tokens)

    @classmethod
    def from_pybrat(
        cls, example: Example, sort_entities: bool = True, sort_relations: bool = True
    ):
        tokens = [
            Token(index=i, start=i, end=i + 1, phrase=char)
            for i, char in enumerate(example.text)
        ]
        entity_map = {
            x.id: Entity(
                type=EntityType(name=x.type),
                token_span=TokenSpan(tokens=tokens[x.start : x.end]),
                phrase=x.mention,
                index=i,
                id=x.id,
            )
            for i, x in enumerate(example.entities)
        }
        relations = [
            Relation(
                type=RelationType(name=x.type),
                head=entity_map[x.arg1.id],
                tail=entity_map[x.arg2.id],
                index=i,
                id=x.id,
            )
            for i, x in enumerate(example.relations)
        ]

        entities = list(entity_map.values())
        if sort_entities:
            entities.sort(key=lambda x: x.start)

        if sort_relations:
            relations.sort(key=lambda x: (x.head.start, x.tail.start))

        return cls(
            tokens=tokens,
            entities=entities,
            relations=relations,
            id=example.id,
            doc=example.text,
        )
