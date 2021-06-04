# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re

from texi.preprocessing import split


class Corpus(object):
    def __init__(self, filename: str) -> None:
        self.filename = filename

    def __iter__(self):
        document = None
        with open(self.filename) as f:
            for line in f:
                line = line.rstrip()
                if not line:
                    continue

                match = re.match(r"^【(?P<title>.+)】$", line)
                if match:
                    if document:
                        yield document

                    document = {
                        "title": match["title"],
                        "sections": [
                            {
                                "section": None,
                                "paragraphs": [],
                            },
                        ],
                    }
                    continue

                match = re.match(
                    r"(?P<section_level>=+)\s*(?P<section>.+)\s*(?P=section_level)",
                    line,
                )
                if match:
                    document["sections"] += [
                        {
                            "section": match["section"],
                            "paragraphs": [],
                        }
                    ]
                    continue

                document["sections"][-1]["paragraphs"] += [split(line, "。")]

            yield document

    def export(self, filename: str, max_length=512) -> None:
        with open(filename, mode="w") as f:
            for document in self:
                for section in document["sections"]:
                    for paragraph in section["paragraphs"]:
                        sentences: list[str] = []
                        length = 0
                        for sentence in paragraph:
                            if length + len(sentence) > max_length:
                                line = "".join(sentences)
                                f.writelines(line + "\n")
                                sentences = []
                                length = 0

                            sentences += [sentence]
                            length += len(sentence)

                        if sentences:
                            line = "".join(sentences)
                            f.writelines(line + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    return parser.parse_args()  # pylint: disable=redefined-outer-name


if __name__ == "__main__":
    args = parse_args()

    corpus = Corpus(args.input)
    corpus.export(args.output)
