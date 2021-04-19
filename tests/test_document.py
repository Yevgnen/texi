# -*- coding: utf-8 -*-

import unittest

from pybrat.parser import Entity as PyBratEntity
from pybrat.parser import Example
from pybrat.parser import Relation as PyBratRelation

from texi.document import (
    Document,
    Entity,
    EntityType,
    Relation,
    RelationType,
    Token,
    TokenSpan,
)


class TestToken(unittest.TestCase):
    def test_init(self):
        index = 0
        start = 1
        end = 7
        phrase = "Python"
        t = Token(index=index, start=start, end=end, phrase=phrase)
        self.assertEqual(t.index, index)
        self.assertEqual(t.start, start)
        self.assertEqual(t.end, end)
        self.assertEqual(len(t), len(phrase))
        for i, char in enumerate(phrase):
            self.assertEqual(t[i], char)


class TestTokenSpan(unittest.TestCase):
    def test_init(self):
        tokens = [
            Token(0, 6, "Python", index=0),
            Token(8, 10, "is", index=1),
            Token(11, 18, "awesome", index=2),
        ]
        token_span = TokenSpan(tokens=tokens)
        self.assertEqual(token_span.start, 0)
        self.assertEqual(token_span.end, 18)
        self.assertEqual(token_span.span, (0, 18))
        self.assertEqual(len(token_span), 3)
        for i, token in enumerate(tokens):
            self.assertEqual(token_span[i], token)


class TestEntityType(unittest.TestCase):
    def test_init(self):
        type_ = EntityType(name="lang")
        self.assertEqual(type_.name, "lang")
        self.assertEqual(str(type_), "lang")


class TestEntity(unittest.TestCase):
    def test_init(self):
        type_ = EntityType(name="person")
        token_span = TokenSpan(
            tokens=[
                Token(index=0, start=0, end=4, phrase="Bill"),
                Token(index=1, start=5, end=10, phrase="Gates"),
            ]
        )
        phrase = "Bill Gates"

        entity = Entity(type=type_, token_span=token_span, phrase=phrase)
        self.assertEqual(entity.type, type_)
        self.assertEqual(entity.token_span, token_span)
        self.assertEqual(entity.phrase, phrase)
        self.assertEqual(entity.start, 0)
        self.assertEqual(entity.end, 10)
        self.assertEqual(entity.span, (0, 10))


class TestRelationType(unittest.TestCase):
    def test_init(self):
        type_ = EntityType(name="is a")
        self.assertEqual(type_.name, "is a")
        self.assertEqual(str(type_), "is a")


class TestRelation(unittest.TestCase):
    def test_init(self):
        Bill = Entity(
            EntityType(name="person"),
            TokenSpan(
                tokens=[
                    Token(index=0, start=0, end=4, phrase="Bill"),
                    Token(index=1, start=5, end=10, phrase="Gates"),
                ]
            ),
            phrase="Bill Gates",
            id="0",
        )
        usa = Entity(
            EntityType(name="location"),
            TokenSpan(
                tokens=[
                    Token(index=5, start=22, end=25, phrase="USA"),
                ]
            ),
            phrase="USA",
            id="1",
        )

        type_ = RelationType(name="born in")
        id_ = "id"
        born_in = Relation(type=type_, head=Bill, tail=usa, id=id_)
        self.assertEqual(born_in.type, type_)
        self.assertEqual(born_in.head, Bill)
        self.assertEqual(born_in.tail, usa)
        self.assertEqual(born_in.id, id_)


class TestDocument(unittest.TestCase):
    def test_init(self):
        Bill = Entity(
            EntityType(name="person"),
            TokenSpan(
                tokens=[
                    Token(index=0, start=0, end=4, phrase="Bill"),
                    Token(index=1, start=5, end=10, phrase="Gates"),
                ]
            ),
            phrase="Bill Gates",
            id="0",
        )
        was = Token(index=1, start=11, end=14, phrase="was")
        born = Token(index=3, start=15, end=19, phrase="born")
        in_ = Token(index=4, start=20, end=22, phrase="in")
        usa = Entity(
            EntityType(name="location"),
            TokenSpan(
                tokens=[
                    Token(index=4, start=22, end=25, phrase="USA"),
                ]
            ),
            phrase="USA",
            id="1",
        )

        type_ = RelationType(name="born in")
        relation_id = "doc_id"
        born_in = Relation(type=type_, head=Bill, tail=usa, id=relation_id)

        doc = "Bill Gates was born in USA."
        tokens = [Bill.token_span[0], was, born, in_, usa.token_span[0]]
        entities = [Bill, usa]
        relations = [born_in]

        doc_id = "doc_id"
        document = Document(
            tokens=tokens, entities=entities, relations=relations, id=doc_id, doc=doc
        )
        self.assertEqual(document.doc, doc)
        self.assertEqual(document.tokens, tokens)
        self.assertEqual(document.entities, entities)
        self.assertEqual(document.relations, relations)
        self.assertEqual(len(document), len(tokens))

    def test_from_pybrat(self):
        pybrat_e1 = PyBratEntity(
            mention="Bill", type="person", start=0, end=4, id="eid1"
        )
        pybrat_e2 = PyBratEntity(
            mention="USA", type="location", start=23, end=26, id="eid2"
        )
        pybrat_relation = PyBratRelation(
            type="born in", arg1=pybrat_e1, arg2=pybrat_e2, id="rid"
        )
        pybrat_entities = [pybrat_e1, pybrat_e2]
        pybrat_relations = [pybrat_relation]
        text = "Bill gates was born in USA."
        example = Example(
            text=text, entities=pybrat_entities, relations=pybrat_relations, id="doc_id"
        )

        doc = Document.from_pybrat(example)
        e1 = Entity(
            type=EntityType(name="person"),
            token_span=TokenSpan(
                tokens=[
                    Token(start=0, end=1, phrase="B", index=0),
                    Token(start=1, end=2, phrase="i", index=1),
                    Token(start=2, end=3, phrase="l", index=2),
                    Token(start=3, end=4, phrase="l", index=3),
                ]
            ),
            phrase="Bill",
            index=0,
            id="eid1",
        )
        e2 = Entity(
            type=EntityType(name="location"),
            token_span=TokenSpan(
                tokens=[
                    Token(start=23, end=24, phrase="U", index=23),
                    Token(start=24, end=25, phrase="S", index=24),
                    Token(start=25, end=26, phrase="A", index=25),
                ]
            ),
            phrase="USA",
            index=1,
            id="eid2",
        )
        relation = Relation(
            type=RelationType("born in"), head=e1, tail=e2, index=0, id="rid"
        )

        for i, token in enumerate(doc.tokens):
            self.assertEqual(token, Token(index=i, start=i, end=i + 1, phrase=text[i]))
        self.assertEqual(doc.entities, [e1, e2])
        self.assertEqual(doc.relations, [relation])
        self.assertEqual(doc.doc, text)
        self.assertEqual(doc.id, "doc_id")


if __name__ == "__main__":
    unittest.main()
