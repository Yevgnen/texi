# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional, Union

from texi.tagger import Tagger
from texi.tagger.sequence_labeling import SequeceLabelingTagger


class DagaTagger(object):
    def __init__(
        self, tagger: Optional[Union[str, SequeceLabelingTagger]] = None
    ) -> None:
        if isinstance(tagger, str):
            tagger = Tagger("iob2")

    def encode(self, Mapping) -> list:
        return

    def decode(self, Sequence) -> dict:
        return
