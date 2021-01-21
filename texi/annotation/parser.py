# -*- coding: utf-8 -*-

import abc
from typing import Any, Dict, Tuple, Union

_Entities = Union[Dict[str, Any], Tuple[str, str, int, int]]


class AnnotationParser(metaclass=abc.ABCMeta):
    def __init__(self, tags=None, relations=None):
        self._tags = tags if tags else {}
        self._relations = relations if relations else {}

    @property
    def tags(self):
        return self._tags

    @property
    def relations(self):
        return self._relations

    @abc.abstractmethod
    def parse_ner(self):
        raise NotImplementedError()
