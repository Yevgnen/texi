# -*- coding: utf-8 -*-

import abc
import itertools
import os

from texi.tagger.sequence_labeling import IOB2


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


class DualFileAnnotationParser(AnnotationParser):
    def __init__(self, **kwargs):
        self.text_ext = kwargs.pop("text_ext")
        self.annotation_ext = kwargs.pop("annotation_ext")
        super().__init__(**kwargs)

    def _iter_dir(self, dirname):
        exts = {self.text_ext, self.annotation_ext}
        for dirpath, _, filenames in os.walk(dirname):
            for filename in filenames:
                if os.path.splitext(filename)[1] in exts:
                    yield os.path.join(dirpath, filename)

    def _iter_dir_samples(self, dirname):
        # Groupby basename before the first '.'.
        for _, pairs in itertools.groupby(
            sorted(self._iter_dir(dirname)),
            key=lambda x: os.path.basename(x).split(".")[0],
        ):

            # Make file pairs (annotation, text).
            pairs = sorted(
                list(pairs), key=lambda x: os.path.splitext(x)[1] == self.text_ext
            )

            # Ignore missing files.
            if len(pairs) == 2:
                yield list(pairs)

    def parse_text(self, filename):
        # Should return a text string.
        raise NotImplementedError()

    def parse_entities(self, filename):
        # Should return iterable in (id, type, start, end, word) format.
        raise NotImplementedError()

    def iter_examples(self, dirname, tagger, error="ignore"):
        keys = ["token", "tag", "start", "end"]

        for ann, txt in self._iter_dir_samples(dirname):
            try:
                example = {
                    "tokens": self.parse_text(txt),
                    "chunks": [
                        dict(zip(keys, [entity, type_, int(start), int(end)]))
                        for (_, type_, start, end, entity) in self.parse_entities(ann)
                    ],
                }
                yield tagger.encode(example)
            except (ValueError, IndexError, TypeError) as e:
                if error == "ignore":
                    continue
                raise e

    def parse_ner(
        self,
        dirname,
        json=False,
        seps="。",
        ignores="\n 　",
        tagger=None,
        join_text=True,
        error="ignore",
    ):
        if tagger is None:
            tagger = IOB2()

        for example in self.iter_examples(dirname, tagger, error=error):
            example_chars, example_tags = [], []
            for i, (char, tag) in enumerate(zip(example["tokens"], example["tags"])):
                if char not in ignores:
                    example_chars += [char]
                    example_tags += [tag]
                if (char in seps or i == len(example["tokens"]) - 1) and example_chars:
                    if join_text:
                        example_chars = "".join(example_chars)

                    if json:
                        yield tagger.decode(
                            {"tokens": example_chars, "tags": example_tags}
                        )
                    else:
                        yield example_chars, example_tags
                    example_chars, example_tags = [], []
