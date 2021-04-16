# -*- coding: utf-8 -*-

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from pybrat.parser import Entity as PyBratEntity
    from pybrat.parser import Example
    from pybrat.parser import Relation as PyBratRelation


@dataclasses.dataclass
class Token(object):
    index: int
    start: int
    end: int
    phrase: str
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
            "id": self.id,
        }

    @classmethod
    def from_pybrat(cls, entity: PyBratEntity):
        tokens = [
            Token(
                index=entity.start + i,
                start=entity.start + i,
                end=entity.start + i + 1,
                phrase=char,
            )
            for i, char in enumerate(entity.mention)
        ]
        token_span = TokenSpan(tokens=tokens)

        return cls(
            type=EntityType(name=entity.type),
            token_span=token_span,
            phrase=entity.mention,
            id=entity.id,
        )


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
    id: Optional[str] = None

    def as_dict(self):
        return {
            "type": str(self.type),
            "head": self.head.as_dict(),
            "tail": self.tail.as_dict(),
            "id": self.id,
        }

    @classmethod
    def from_pybrat(cls, relation: PyBratRelation):
        head = Entity.from_pybrat(relation.arg1)
        tail = Entity.from_pybrat(relation.arg2)

        return cls(
            type=RelationType(name=relation.type), head=head, tail=tail, id=relation.id
        )


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
    def from_pybrat(cls, example: Example):
        tokens = [
            Token(index=i, start=i, end=i + 1, phrase=char)
            for i, char in enumerate(example.text)
        ]
        entities = [Entity.from_pybrat(x) for x in example.entities]
        relations = [Relation.from_pybrat(x) for x in example.relations]

        return cls(
            tokens=tokens,
            entities=entities,
            relations=relations,
            id=example.id,
            doc=example.text,
        )
