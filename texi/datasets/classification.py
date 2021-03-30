# -*- coding: utf-8 -*-

import glob
import json
import os
import zipfile

import pandas as pd

from texi.datasets.dataset import Datasets


class CHIP2019(Datasets):
    @classmethod
    def from_dir(cls, dirname: str):
        prefix = "CHIP2019"
        df_train = pd.read_csv(os.path.join(dirname, "train.csv"))
        df_train["id"] = [f"{prefix}_train_{i}" for i in range(len(df_train))]
        df_val = pd.read_csv(os.path.join(dirname, "dev_id.csv"))
        df_val["id"] = df_val["id"].map(lambda x: f"{prefix}_val_{x}")

        new_names = {"question1": "sentence1", "question2": "sentence2"}
        df_train.rename(new_names, axis=1, inplace=True)
        df_val.rename(new_names, axis=1, inplace=True)

        return cls(
            train=df_train.to_dict("records"),
            val=df_val.to_dict("records"),
            dirname=dirname,
        )


class NCOV2019(Datasets):
    @classmethod
    def from_dir(cls, dirname: str):
        def _load_data(basename, mode):
            df = pd.read_csv(os.path.join(dirname, f"{basename}.csv"))
            df["id"] = df["id"].map(lambda x: f"{prefix}_{mode}_{x}")
            df.rename(
                {"query1": "sentence1", "query2": "sentence2"}, axis=1, inplace=True
            )

            return df.to_dict("records")

        prefix = "NCOV2019"

        return cls(
            train=_load_data("train", "train"),
            val=_load_data("dev", "val"),
            dirname=dirname,
        )


class LUGETextPairDataset(Datasets):
    @classmethod
    def from_dir(cls, dirname: str):
        def _load_data(basename, mode):
            columns = ["sentence1", "sentence2"]
            if mode != "test":
                columns += ["label"]

            df = pd.read_csv(
                os.path.join(dirname, f"{basename}.tsv"),
                sep="\t",
                names=columns,
            )
            df["id"] = df.index.map(lambda x: f"{prefix}_{mode}_{x}")

            return df.to_dict("records")

        prefix = "NCOV2019"

        return cls(
            train=_load_data("train", "train"),
            val=_load_data("dev", "val"),
            test=_load_data("test", "test"),
            dirname=dirname,
        )


class LCQMC(LUGETextPairDataset):
    pass


class BQCorpus(LUGETextPairDataset):
    pass


class PAWSX(LUGETextPairDataset):
    pass


class AFQMC(Datasets):
    @classmethod
    def from_dir(cls, dirname: str):
        def _load_data(basename, mode):
            df = pd.read_csv(
                os.path.join(dirname, f"{basename}.csv"),
                sep="\t",
                names=["id", "sentence1", "sentence2", "label"],
            )
            df["id"] = df["id"].map(lambda x: f"AFQMC_{mode}_{x}")

            return df.to_dict("records")

        train1 = _load_data("atec_nlp_sim_train", "train")
        train2 = _load_data("atec_nlp_sim_train_add", "train")

        return cls(train=train1 + train2, dirname=dirname)


class CCKS2018(Datasets):
    @classmethod
    def from_dir(cls, dirname: str):
        def _load_data(basename, mode):
            columns = ["sentence1", "sentence2"]
            if mode != "train":
                columns = ["id"] + columns
            else:
                columns += ["label"]

            df = pd.read_csv(
                os.path.join(dirname, f"{basename}.txt"),
                sep="\t",
                names=columns,
            )

            if mode == "train":
                df["id"] = df.index.map(lambda x: f"CCKS2018_{mode}_{x}")

            return df.to_dict("records")

        train = _load_data("task3_train", "train")
        val = _load_data("task3_dev", "val")

        test_zip = os.path.join(dirname, "task3_test_data_expand.zip")
        with zipfile.ZipFile(test_zip) as zip:
            with zip.open("task3_test_data_expand/test_with_id.txt", mode="r") as f:
                test = pd.read_csv(f, sep="\t", names=["id", "sentence1", "sentence2"])
                test = test.to_dict("records")

        return cls(train=train, val=val, test=test, dirname=dirname)


class ChineseSNLI(Datasets):
    @classmethod
    def from_dir(cls, dirname: str):
        def _load_data(basename, mode):
            filename = os.path.join(dirname, f"cnsd_snli_v1.0.{basename}.jsonl")
            with open(filename, mode="r") as f:
                examples = []
                for i, line in enumerate(f):
                    example = json.loads(line.rstrip())
                    example["id"] = f"ChineseSNLI_{mode}_{i}"
                    example["label"] = example.pop("gold_label")
                    examples += [example]

                return examples

        return cls(
            train=_load_data("train", "train"),
            val=_load_data("dev", "val"),
            test=_load_data("test", "test"),
            dirname=dirname,
        )


class ChineseSTSB(Datasets):
    @classmethod
    def from_dir(cls, dirname: str):
        def _load_data(basename, mode):
            filename = os.path.join(dirname, f"cnsd-sts-{basename}.txt")
            columns = ["id", "sentence1", "sentence2", "label"]
            with open(filename, mode="r") as f:
                examples = []
                for line in f:
                    id_, sentence1, sentence2, label = line.rstrip().split("||")
                    label = int(label)
                    examples += [dict(zip(columns, [id_, sentence1, sentence2, label]))]

                return examples

        return cls(
            train=_load_data("train", "train"),
            val=_load_data("dev", "val"),
            test=_load_data("test", "test"),
            dirname=dirname,
        )


class THUCNews(Datasets):
    @classmethod
    def from_dir(cls, dirname: str):
        records = []
        for filename in glob.iglob(os.path.join(dirname, "**/*.txt"), recursive=True):
            relname = os.path.relpath(filename, dirname)
            label = os.path.dirname(relname)
            id_ = os.path.splitext(os.path.basename(relname))[0]
            with open(filename) as f:
                text = f.read(filename)
            records += [{"id": id_, "text": text, "label": label}]

        return cls(train=records, dirname=dirname)
