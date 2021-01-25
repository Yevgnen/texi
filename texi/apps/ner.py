# -*- coding: utf-8 -*-

import collections
import itertools
import os
import re
import subprocess
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import ahocorasick
from felis.random import random_colors
from felis.palette import Colors

from texi.tagger import Tagger


class NERDataReport(object):
    def __init__(self, examples: Sequence[Mapping], seps: str = "。", k: int = 10):
        # TODO: 2019-09-27 Accept list of tokens inputs of `text` field.
        self.examples = list(examples)
        self.seps = seps
        self.k = k

    def _group_entities(self, entities, unique=True):
        agg = set if unique else list

        return [
            (key, agg(x[1] for x in agg(group)))
            for key, group in itertools.groupby(
                sorted(entities, key=lambda x: x[0]), key=lambda x: x[0]
            )
        ]

    def _split_sentence(self, text):
        if not self.seps:
            return [text]

        text = re.sub(rf"{re.escape(self.seps)}$", "", text)
        sentences = re.split(rf"{re.escape(self.seps)}", text)

        return sentences

    @property
    def entities(self) -> List[Tuple[str, str]]:
        return [
            (x["tag"], x["token"]) for sample in self.examples for x in sample["chunks"]
        ]

    @property
    def sentences(self) -> List[str]:
        return [
            *itertools.chain.from_iterable(
                self._split_sentence(x["tokens"]) for x in self.examples
            )
        ]

    @property
    def sample_size(self) -> int:
        return len(self.examples)

    @property
    def annotation_consistency(self) -> float:
        passes = 0
        for example in self.examples:
            text = example["tokens"]
            for entity in example["chunks"]:
                if text[entity["start"] : entity["end"]] != entity["token"]:
                    break
            else:
                passes += 1
        consistency = passes / len(self.examples)

        return consistency

    @property
    def annotation_rate(self) -> float:
        def _tag(text, entities):
            tags = ["O"] * len(text)
            for entity in entities:
                tags[entity["start"] : entity["end"]] = ["N"] * (
                    entity["end"] - entity["start"]
                )

            return tags

        def _split(text, tags):
            split_tags = []
            sample_tags = []
            for i, (char, tag) in enumerate(zip(text, tags)):
                sample_tags += [tag]
                if (char in self.seps or i == len(text) - 1) and sample_tags:
                    split_tags += [sample_tags]
                    sample_tags = []

            return split_tags

        tagged = [(x["tokens"], _tag(x["tokens"], x["chunks"])) for x in self.examples]
        tags = [*itertools.chain.from_iterable(_split(*x) for x in tagged)]

        annotated = {i for i, x in enumerate(tags) if set(x) == {"O"}}
        annotation_rate = len(annotated) / len(tags)

        return annotation_rate

    @property
    def annotation_missing_rate(self) -> float:
        actree = ahocorasick.Automaton()
        counts = {x["token"]: 0 for example in self.examples for x in example["chunks"]}
        for word in counts.keys():
            actree.add_word(word, word)
        actree.make_automaton()

        for example in self.examples:
            text = example["tokens"]
            entities = list(
                sorted(example["chunks"], key=lambda x: (x["start"], x["end"]))
            )
            for i, entity in enumerate(entities):
                if i == 0:
                    span = text[: entity["start"]]
                elif i == len(entities) - 1:
                    span = text[entity["end"] :]
                else:
                    span = text[entities[i - 1]["end"] : entity["start"]]

                for _, match in actree.iter(span):
                    counts[match] += 1

        missing_counts = len(
            {i for i, count in enumerate(counts.values()) if count > 0}
        )
        missing_rate = missing_counts / len(counts)

        return missing_rate

    @property
    def text_length(self) -> float:
        average_length = sum(len(x["tokens"]) for x in self.examples) / len(
            self.examples
        )

        return average_length

    @property
    def sentence_average_length(self) -> float:
        average_length = sum(len(x) for x in self.sentences) / len(self.sentences)

        return average_length

    @property
    def sentence_average_count(self) -> float:
        average_count = len(self.sentences) / len(self.examples)

        return average_count

    @property
    def sentence_overlapping_rate(self) -> float:
        counter = collections.Counter(self.sentences)
        overlaps = {key: value for key, value in counter.items() if value > 1}
        overlapping_rate = len(overlaps) / len(counter)

        return overlapping_rate

    @property
    def entity_singleton_size(self) -> int:
        counter = collections.Counter(self.entities)
        singletons = {key for key, value in counter.items() if value < 2}

        return len(singletons)

    @property
    def entity_imbalance(self) -> float:
        dists = dict(collections.Counter([key for key, _ in self.entities]))
        total = sum(dists.values())
        dists = {key: value / total for key, value in dists.items()}
        imbalance = max(dists.values()) / min(dists.values())

        return imbalance

    @property
    def entity_size(self) -> Dict[str, int]:
        entities = self._group_entities(self.entities, unique=False)
        sizes = {key: len(value) for key, value in entities}

        return sizes

    @property
    def entity_frequence(self) -> Dict[str, int]:
        entities = self._group_entities(self.entities, unique=False)
        counters = {key: collections.Counter(value) for key, value in entities}
        freqs = {
            key: sum(value.values()) / len(value) for key, value in counters.items()
        }

        return freqs

    @property
    def entity_uniques(self) -> Dict[str, int]:
        entities = self._group_entities(self.entities, unique=True)
        uniques = {key: len(value) for key, value in entities}

        return uniques

    @property
    def entity_overlapping_rate(self) -> Dict[Tuple[str, str], float]:
        entities = self._group_entities(self.entities, unique=True)
        overlaps = {}
        for k1, v1 in entities:
            for k2, v2 in entities:
                overlaps[(k1, k2)] = len(v1 & v2) / min(len(v1), len(v2))

        return overlaps

    @property
    def top_entities(self) -> Dict[str, Tuple[str, int, float, float]]:
        def _top_k(entities):
            counter = collections.Counter(entities)
            total = sum(counter.values())
            top_k = counter.most_common(self.k)

            return [x + (x[1] / total, x[1] / len(counter)) for x in top_k]

        return {
            type_: _top_k(entities)
            for type_, entities in self._group_entities(self.entities, unique=False)
        }

    def describe(self) -> Dict[str, Any]:
        indicators = [
            "sample_size",
            "annotation_consistency",
            "annotation_rate",
            "annotation_missing_rate",
            "text_length",
            "sentence_average_count",
            "sentence_average_length",
            "sentence_overlapping_rate",
            "entity_singleton_size",
            "entity_imbalance",
            "entity_size",
            "entity_uniques",
            "entity_frequence",
            "entity_overlapping_rate",
            "top_entities",
        ]

        return {indicator: getattr(self, indicator) for indicator in indicators}

    def to_html(self, filename: str) -> None:
        # pylint: disable=import-outside-toplevel
        import pandas as pd
        import plotly.figure_factory as ff
        import plotly.io as io

        border = pd.options.display.html.border
        pd.options.display.html.border = 0

        report = self.describe()

        # Overview
        body = ["<h1>Overview</h1>"]
        overview_index = [
            "sample_size",
            "annotation_consistency",
            "annotation_rate",
            "annotation_missing_rate",
            "text_length",
            "sentence_average_count",
            "sentence_average_length",
            "sentence_overlapping_rate",
            "entity_singleton_size",
            "entity_imbalance",
        ]
        overview = pd.Series(
            {k: report[k] for k in overview_index}, index=overview_index, name="value"
        )
        body += [overview.to_frame().to_html()]

        body += ["<h1>Entity</h1>"]
        entity_report_columns = ["entity_size", "entity_uniques", "entity_frequence"]
        entity_report = pd.DataFrame(
            {k: report[k] for k in entity_report_columns}, columns=entity_report_columns
        )
        entity_report.loc["#SUM"] = entity_report.sum(axis=0)
        body += [entity_report.to_html()]
        body += ["<h2>Overlapping</h2>"]
        entity_overlapping_rate = (
            pd.Series(
                report["entity_overlapping_rate"].values(),
                index=pd.MultiIndex.from_tuples(
                    report["entity_overlapping_rate"].keys()
                ),
            )
            .round(2)
            .unstack()
        )
        fig_entity_overlapping_rate = ff.create_annotated_heatmap(
            z=entity_overlapping_rate.values,
            x=entity_overlapping_rate.index.tolist(),
            y=entity_overlapping_rate.columns.tolist(),
            zmin=0,
            zmax=1,
            showscale=True,
            colorscale="Blues",
        )
        body += [io.to_html(fig_entity_overlapping_rate)]

        body += ["<h2>Top Entities</h2>"]
        for type_, top_k in report["top_entities"].items():
            body += [f"<h3>{type_}</h3>"]
            df_top = pd.DataFrame.from_records(
                top_k, columns=["entity", "count", "frequency", "frequency-unique"]
            ).set_index("entity")
            df_top.loc["#SUM"] = df_top.sum(axis=0)
            body += [df_top.to_html()]

        title = "NER Report"
        body = "".join(body)
        html = f"""<html>
        <head>
            <title>{title}</title>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/kognise/water.css@latest/dist/light.min.css">
        </head>
        <body>{body}</body>
    </html>"""

        with open(filename, mode="w") as f:
            f.writelines(html)

        pd.options.display.html.border = border


def _spacy_visual_ner(examples, filename, colors):
    # pylint: disable=import-outside-toplevel
    from spacy import displacy

    # Generate spacy inputs.
    spacy_data = [
        {
            "tokens": example["tokens"],
            "ents": [
                {"start": x["start"], "end": x["end"], "label": x["tag"]}
                for x in example["chunks"]
            ],
            "title": example.get("id", f"#{i}"),
        }
        for i, example in enumerate(examples)
    ]

    # Give each type a color.
    if not colors:
        types = set(x["tag"] for example in examples for x in example["chunks"])
        colors = random_colors(Colors.PRESETS, num=len(types))
        colors = dict(zip(types, colors))

    display_fn = displacy.render if filename else displacy.serve
    rendered = display_fn(
        spacy_data, style="ent", manual=True, options={"colors": colors}, page=True
    )

    if filename:
        with open(filename, mode="w") as f:
            f.writelines(rendered)


def visualize_ner(
    examples: Sequence[Mapping],
    filename: Optional[str] = None,
    colors: Optional[Mapping[str, str]] = None,
    sort: bool = False,
    drop_duplicates: bool = False,
) -> None:
    if drop_duplicates:
        tagger = Tagger("iob2")
        examples = [
            dict(x)
            for x in {
                (("tokens", x["tokens"]), ("tag", tuple(x["tag"])))
                for x in map(tagger.encode, examples)
            }
        ]
        examples = [*map(tagger.decode, examples)]

    if sort:
        examples = sorted(examples, key=lambda x: len(x["tokens"]))

    _spacy_visual_ner(examples, filename, colors)


def visualize_ner_prediction(
    y: Sequence[Mapping],
    y_pred: Sequence[Mapping],
    filename: Optional[str] = None,
    colors: Optional[Mapping[str, str]] = None,
    ignore_corrections: bool = False,
) -> None:
    if ignore_corrections:
        y, y_pred = zip(*[(yi, yi_pred) for yi, yi_pred in zip(y, y_pred)])

    data = itertools.chain.from_iterable(zip(y, y_pred))

    _spacy_visual_ner(data, filename, colors)


def merge_ner_splits(splits: Sequence[str], entities: Sequence[Dict]) -> List[Dict]:
    if len(splits) != len(entities):
        raise ValueError("`splits` and `entities` have different lengths")

    merged = []
    offset = 0
    for split, entities_of_split in zip(splits, entities):
        for entity in entities_of_split:
            entity["start"] += offset
            entity["end"] += offset
            merged += [entity]
        offset += len(split)

    return merged


def conlleval_file(filename: str) -> Dict:
    script = os.path.join(os.path.dirname(__file__), "conlleval.pl")
    with open(filename) as f:
        outputs = subprocess.run(
            [script, "-d", "\t"], text=True, stdin=f, capture_output=True, check=True
        ).stdout
        # Parse outputs
        results = {"tags": {}}

        lines = []
        for line in outputs.split("\n"):
            line = line.strip()
            if line:
                lines += [line]

        for i, line in enumerate(lines):
            if i == 0:
                results["stats"] = {
                    key: float(value)
                    for key, value in zip(
                        ["tokens", "phrases", "found", "correct"],
                        re.findall(r"[0-9.]", line),
                    )
                }
            elif i == 1:
                results["metrics"] = {
                    key if key != "FB1" else "f1": float(value) / 100.0
                    for key, value in zip(
                        ["accuracy", "precision", "recall", "FB1"],
                        re.findall(r"(?<=\s)[0-9.]+", line),
                    )
                }
            else:
                tag, *metrics, support = re.split(r"(?:%;|:)?\s+", line)
                results["tags"][tag] = {
                    metrics[i]
                    if metrics[i] != "FB1"
                    else "f1": float(metrics[i + 1]) / 100.0
                    for i in range(0, len(metrics), 2)
                }
                results["tags"][tag]["support"] = float(support)

        return results


def conlleval(data: Iterable[Iterable[str]], filename: str = None) -> Dict:
    with (
        (filename and open(filename, mode="w"))
        or tempfile.NamedTemporaryFile(mode="w", delete=False)
    ) as f:
        for example in data:
            for row in zip(*example):
                f.writelines("\t".join(row))
                f.writelines("\n")
            f.writelines("\n")

    return conlleval_file(f.name)


def classification_report(y_true, y_pred, **kwargs):
    # pylint: disable=import-outside-toplevel
    from sklearn.metrics import classification_report as sklearn_classification_report
    from sklearn.preprocessing import LabelBinarizer

    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(itertools.chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(itertools.chain.from_iterable(y_pred)))

    tagset = set(lb.classes_) - {"O"}
    tagset = sorted(tagset, key=lambda tag: tag.split("-", 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    return sklearn_classification_report(
        y_true_combined,
        y_pred_combined,
        labels=[class_indices[cls] for cls in tagset],
        target_names=tagset,
        **kwargs,
    )


def confusion_matrix(y_true, y_pred):
    # pylint: disable=import-outside-toplevel
    import pandas as pd
    from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

    y_true = [*itertools.chain.from_iterable(y_true)]
    y_pred = [*itertools.chain.from_iterable(y_pred)]

    tags = sorted(
        list(set(y_true)),
        key=lambda x: tuple(reversed(x.split("-"))) if "-" in x else (x,),
    )

    return pd.DataFrame(
        sklearn_confusion_matrix(y_true, y_pred, labels=tags), index=tags, columns=tags
    )
