# -*- coding: utf-8 -*-

import os


class CHIP2019(object):
    def __init__(self, dirname):
        self.dirname = dirname
        self._load_data()

    def _load_data(self):
        # pylint: disable=import-outside-toplevel
        import pandas as pd

        prefix = "CHIP2019"
        df_train = pd.read_csv(os.path.join(self.dirname, "train.csv"))
        df_train["id"] = [f"{prefix}_train_{i}" for i in range(len(df_train))]
        df_val = pd.read_csv(os.path.join(self.dirname, "dev_id.csv"))
        df_val["id"] = df_val["id"].map(lambda x: f"{prefix}_val_{x}")

        new_names = {"question1": "query", "question2": "doc"}
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
            # pylint: disable=import-outside-toplevel
            import pandas as pd

            df = pd.read_csv(os.path.join(self.dirname, f"{basename}.csv"))
            df["id"] = df["id"].map(lambda x: f"{prefix}_{mode}_{x}")
            df.rename({"query1": "query", "query2": "doc"}, axis=1, inplace=True)

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
            # pylint: disable=import-outside-toplevel
            import pandas as pd

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
