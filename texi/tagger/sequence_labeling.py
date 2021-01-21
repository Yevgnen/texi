# -*- coding: utf-8 -*-

import abc
import collections
import itertools
from typing import Dict, Iterable, List, Mapping


class SequeceLabelingTagger(metaclass=abc.ABCMeta):
    def _iter_chunks(self, chunks):
        for chunk in chunks:
            if isinstance(chunk, collections.abc.Mapping):
                token, tag, start, end = (
                    chunk["token"],
                    chunk["tag"],
                    int(chunk["start"]),
                    int(chunk["end"]),
                )
            else:
                token, tag, start, end = chunk
                start, end = int(start), int(end)

            yield token, tag, start, end

    @abc.abstractmethod
    def encode(self, inputs: Mapping) -> Dict:
        raise NotImplementedError()

    @abc.abstractmethod
    def decode(self, inputs: Mapping) -> Dict:
        raise NotImplementedError()

    @staticmethod
    def from_text(filename: str, sep: str = "\t") -> List[Dict]:
        examples, example = [], []
        with open(filename) as f:
            for line in f:
                line = line.rstrip()

                if not line:
                    if example:
                        tokens, chunks = zip(*example)
                        examples += [{"tokens": list(tokens), "tags": list(chunks)}]
                        example = []
                    continue

                example += [line.split(sep)]

        return examples

    @classmethod
    def to_text(
        cls, filename: str, examples: Iterable[Mapping], sep: str = "\t"
    ) -> None:
        with open(filename, mode="w") as f:
            for example in examples:
                f.writelines(
                    itertools.chain(
                        f"{token}{sep}{tag}\n"
                        for token, tag in zip(example["tokens"], example["tags"])
                    )
                )
                f.writelines("\n")


class IOB1(SequeceLabelingTagger):
    def encode(self, inputs: Mapping) -> Dict:
        tokens, chunks = inputs["tokens"], inputs["chunks"]

        tags = ["O"] * len(tokens)
        for token, tag, start, end in self._iter_chunks(chunks):
            assert token == tokens[start:end]

            I_tag, B_tag = f"I-{tag}", f"B-{tag}"
            tags[start:end] = [I_tag] * (end - start)
            if start > 0 and tags[start - 1] in {I_tag, B_tag}:
                tags[start] = f"B-{tag}"

        return {"tokens": tokens, "tags": tags}

    def decode(self, inputs: Mapping) -> Dict:
        tokens, tags = inputs["tokens"], inputs["tags"]

        chunks = []
        start = -1
        current_tag = None
        for i, tag in enumerate(tags):
            if tag == "O":
                prefix, tag = tag, None
            else:
                prefix, tag = tag.split("-")

            if prefix == "I" and tag == current_tag:
                continue

            if current_tag and start >= 0:
                chunks += [
                    {
                        "token": tokens[start:i],
                        "tag": current_tag,
                        "start": start,
                        "end": i,
                    }
                ]
                start = -1
                current_tag = None

            if prefix == "B" or prefix == "I" and not current_tag or tag != current_tag:
                start = i
                current_tag = tag

        if prefix != "O":
            chunks += [
                {
                    "token": tokens[start:],
                    "tag": tag,
                    "start": start,
                    "end": len(tokens),
                }
            ]

        return {"tokens": tokens, "chunks": chunks}


class IOB2(SequeceLabelingTagger):
    def encode(self, inputs: Mapping) -> Dict:
        tokens, chunks = inputs["tokens"], inputs["chunks"]

        tags = ["O"] * len(tokens)
        for token, tag, start, end in self._iter_chunks(chunks):
            assert token == tokens[start:end]

            tags[start] = f"B-{tag}"
            tags[start + 1 : end] = [f"I-{tag}"] * (end - start - 1)

        return {"tokens": tokens, "tags": tags}

    def decode(self, inputs: Mapping) -> Dict:
        tokens, tags = inputs["tokens"], inputs["tags"]

        chunks = []
        start = -1
        current_tag = None
        for i, tag in enumerate(tags):
            if tag == "O":
                prefix, tag = tag, None
            else:
                prefix, tag = tag.split("-")

            if prefix == "I" and tag == current_tag:
                continue

            if current_tag and start >= 0:
                chunks += [
                    {
                        "token": tokens[start:i],
                        "tag": current_tag,
                        "start": start,
                        "end": i,
                    }
                ]
                start = -1
                current_tag = None

            if prefix == "B":
                start = i
                current_tag = tag

        if prefix != "O":
            chunks += [
                {
                    "token": tokens[start:],
                    "tag": tag,
                    "start": start,
                    "end": len(tokens),
                }
            ]

        return {"tokens": tokens, "chunks": chunks}


class IOBES(SequeceLabelingTagger):
    def encode(self, inputs: Mapping) -> Dict:
        tokens, chunks = inputs["tokens"], inputs["chunks"]

        tags = ["O"] * len(tokens)
        for token, tag, start, end in self._iter_chunks(chunks):
            assert token == tokens[start:end]

            if start + 1 == end:
                tags[start] = f"S-{tag}"
            else:
                tags[start] = f"B-{tag}"
                tags[start + 1 : end - 1] = [f"I-{tag}"] * (end - start - 2)
                tags[end - 1] = f"E-{tag}"

        return {"tokens": tokens, "tags": tags}

    def decode(self, inputs: Mapping) -> Dict:
        tokens, tags = inputs["tokens"], inputs["tags"]

        chunks = []
        start = -1
        current_tag = None
        for i, tag in enumerate(tags):
            if tag == "O":
                prefix, tag = tag, None
            else:
                prefix, tag = tag.split("-")

            if prefix == "S":
                chunks += [
                    {"token": tokens[i : i + 1], "tag": tag, "start": i, "end": i + 1}
                ]
                start = -1
                current_tag = None
                continue

            if prefix == "I":
                if tag == current_tag:
                    continue

                start = -1
                current_tag = None

            if prefix == "E":
                if current_tag and start >= 0 and tag == current_tag:
                    chunks += [
                        {
                            "token": tokens[start : i + 1],
                            "tag": current_tag,
                            "start": start,
                            "end": i + 1,
                        }
                    ]
                start = -1
                current_tag = None

            if prefix == "B":
                start = i
                current_tag = tag

        return {"tokens": tokens, "chunks": chunks}
