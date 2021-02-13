# -*- coding: utf-8 -*-

import json
import os
import zipfile

import pandas as pd


class CHIP2019(object):
    def __init__(self, dirname):
        self.dirname = dirname
        self._load_data()

    def _load_data(self):
        prefix = "CHIP2019"
        df_train = pd.read_csv(os.path.join(self.dirname, "train.csv"))
        df_train["id"] = [f"{prefix}_train_{i}" for i in range(len(df_train))]
        df_val = pd.read_csv(os.path.join(self.dirname, "dev_id.csv"))
        df_val["id"] = df_val["id"].map(lambda x: f"{prefix}_val_{x}")

        new_names = {"question1": "sentence1", "question2": "sentence2"}
        df_train.rename(new_names, axis=1, inplace=True)
        df_val.rename(new_names, axis=1, inplace=True)

        self.train = df_train.to_dict("records")
        self.val = df_val.to_dict("records")


class NCOV2019(object):
    def __init__(self, dirname):
        self.dirname = dirname
        self._load_data()

    def _load_data(self):
        def _load_data(basename, mode):
            df = pd.read_csv(os.path.join(self.dirname, f"{basename}.csv"))
            df["id"] = df["id"].map(lambda x: f"{prefix}_{mode}_{x}")
            df.rename(
                {"query1": "sentence1", "query2": "sentence2"}, axis=1, inplace=True
            )

            return df.to_dict("records")

        prefix = "NCOV2019"
        self.train = _load_data("train", "train")
        self.val = _load_data("dev", "val")


class LUGETextPairDataset(object):
    def __init__(self, dirname):
        self.dirname = dirname
        self._load_data()

    def _load_data(self):
        def _load_data(basename, mode):
            columns = ["sentence1", "sentence2"]
            if mode != "test":
                columns += ["label"]

            df = pd.read_csv(
                os.path.join(self.dirname, f"{basename}.tsv"),
                sep="\t",
                names=columns,
            )
            df["id"] = df.index.map(lambda x: f"{prefix}_{mode}_{x}")

            return df.to_dict("records")

        prefix = "NCOV2019"
        self.train = _load_data("train", "train")
        self.val = _load_data("dev", "val")
        self.test = _load_data("test", "test")


class LCQMC(LUGETextPairDataset):
    pass


class BQCorpus(LUGETextPairDataset):
    pass


class PAWSX(LUGETextPairDataset):
    pass


class AFQMC(object):
    def __init__(self, dirname):
        self.dirname = dirname
        self._load_data()

    def _load_data(self):
        def _load_data(basename, mode):
            df = pd.read_csv(
                os.path.join(self.dirname, f"{basename}.csv"),
                sep="\t",
                names=["id", "sentence1", "sentence2", "label"],
            )
            df["id"] = df["id"].map(lambda x: f"AFQMC_{mode}_{x}")

            return df.to_dict("records")

        train1 = _load_data("atec_nlp_sim_train", "train")
        train2 = _load_data("atec_nlp_sim_train_add", "train")
        self.train = train1 + train2


class CCKS2018(object):
    def __init__(self, dirname):
        self.dirname = dirname
        self._load_data()

    def _load_data(self):
        def _load_data(basename, mode):
            columns = ["sentence1", "sentence2"]
            if mode != "train":
                columns = ["id"] + columns
            else:
                columns += ["label"]

            df = pd.read_csv(
                os.path.join(self.dirname, f"{basename}.txt"),
                sep="\t",
                names=columns,
            )

            if mode == "train":
                df["id"] = df.index.map(lambda x: f"CCKS2018_{mode}_{x}")

            return df.to_dict("records")

        self.train = _load_data("task3_train", "train")
        self.val = _load_data("task3_dev", "val")

        test_zip = os.path.join(self.dirname, "task3_test_data_expand.zip")
        with zipfile.ZipFile(test_zip) as zip:
            with zip.open("task3_test_data_expand/test_with_id.txt", mode="r") as f:
                test = pd.read_csv(f, sep="\t", names=["id", "sentence1", "sentence2"])
                self.test = test.to_dict("records")


class ChineseSNLI(object):
    def __init__(self, dirname):
        self.dirname = dirname
        self._load_data()

    def _load_data(self):
        def _load_data(basename, mode):
            filename = os.path.join(self.dirname, f"cnsd_snli_v1.0.{basename}.jsonl")
            with open(filename, mode="r") as f:
                examples = []
                for i, line in enumerate(f):
                    example = json.loads(line.rstrip())
                    example["id"] = f"ChineseSNLI_{mode}_{i}"
                    example["label"] = example.pop("gold_label")
                    examples += [example]

                return examples

        self.train = _load_data("train", "train")
        self.val = _load_data("dev", "val")
        self.test = _load_data("test", "test")


class ChineseSTSB(object):
    def __init__(self, dirname):
        self.dirname = dirname
        self._load_data()

    def _load_data(self):
        def _load_data(basename, mode):
            filename = os.path.join(self.dirname, f"cnsd-sts-{basename}.txt")
            columns = ["id", "sentence1", "sentence2", "label"]
            with open(filename, mode="r") as f:
                examples = []
                for line in f:
                    id_, sentence1, sentence2, label = line.rstrip().split("||")
                    label = int(label)
                    examples += [dict(zip(columns, [id_, sentence1, sentence2, label]))]

                return examples

        self.train = _load_data("train", "train")
        self.val = _load_data("dev", "val")
        self.test = _load_data("test", "test")
