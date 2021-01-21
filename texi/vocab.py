# -*- coding: utf-8 -*-

import collections
import itertools

import numpy as np


class SpecialTokens(object):
    def __init__(self, pad="[PAD]", unk="[UNK]", bos="[BOS]", eos="[EOS]"):
        self.pad = pad
        self.unk = unk
        self.bos = bos
        self.eos = eos


class Vocabulary(object):
    def __init__(
        self,
        docs=None,
        min_count=1,
        max_size=float("inf"),
        keep_case=False,
        specials=None,
        default=None,
    ):
        self.keep_case = keep_case
        if specials is None:
            specials = []
        if default is not None:
            specials += [default]
        if not self.keep_case:
            specials = [x.lower() for x in specials]
        self.specials = collections.OrderedDict.fromkeys(specials)
        self.default = default
        self.reset()

        if docs:
            self.learn(docs)
            self.trim(min_count=min_count, max_size=max_size)

    @property
    def words(self):
        return set(self.freqs.keys())

    def _preprocess(self, word):
        if not self.keep_case:
            word = word.lower()

        return word

    def reset(self):
        self.freqs = collections.defaultdict(
            int, {w: float("inf") for w in self.specials}
        )
        self.word2index = {w: i for i, w in enumerate(self.freqs)}
        self.index2word = {i: w for w, i in self.word2index.items()}

    def add(self, word, freq=1):
        word = self._preprocess(word)

        if word in self.specials:
            raise ValueError("can not add special token")

        if word not in self.freqs:
            self.word2index[word] = len(self)
            self.index2word[len(self)] = word
        self.freqs[word] += freq

    def learn(self, docs):
        if isinstance(docs, str):
            self.add(docs)
            return

        if not isinstance(docs[0], str):
            docs = itertools.chain(*docs)

        for word in docs:
            self.add(word)

    def compactify(self):
        freqs = {w: f for w, f in self.freqs.items() if w not in self.specials}
        self.reset()
        offset = len(self.freqs)
        for i, (w, f) in enumerate(freqs.items()):
            self.word2index.update({w: i + offset})
            self.index2word.update({i + offset: w})
            self.freqs.update({w: f})

    def trim(self, min_count=None, max_size=None, compactify=False):
        if min_count:
            self.freqs = collections.defaultdict(
                int, {k: v for k, v in self.freqs.items() if v >= min_count}
            )

        if max_size is not None:
            if max_size < len(self.specials):
                raise ValueError("`max_size` too small")
            if len(self) > max_size:
                self.freqs = collections.defaultdict(
                    int, sorted(self.freqs.items(), key=lambda x: -x[1])[:max_size]
                )

        if compactify:
            self.compactify()

    def save(self, filename):
        with open(filename, mode="w") as f:
            f.writelines(
                "\n".join(
                    [
                        f"{word}\t{freq}"
                        for word, freq in self.freqs.items()
                        if word not in set(self.specials)
                    ]
                )
            )

    def load(self, filename):
        self.reset()

        with open(filename) as f:
            for line in f:
                line = line.strip()
                if line:
                    word, freq = line.split("")
                    if word in freq:
                        raise ValueError(f"Duplicate key: {word!r}")
                    self.add(word, float(freq))

    def get_index(self, word):
        word = self._preprocess(word)
        index = self.word2index.get(word)

        if index is not None:
            return index

        if self.default is not None:
            return self.word2index.get(self.default)

        raise KeyError(f"Word not found while `default` is not set: {word}")

    def get_word(self, index):
        return self.index2word[index]

    def transform(self, words):
        if isinstance(words, str):
            return self.get_index(words)

        return [self.transform(word) for word in words]

    def inverse_transform(self, ids):
        if isinstance(ids, (int, np.integer)):
            return self.get_word(ids)

        return [self.inverse_transform(x) for x in ids]

    def __getitem__(self, key):
        return self.get_index(key)

    def __len__(self):
        return len(self.freqs)

    def __contains__(self, key):
        return key in self.freqs.keys()

    def __delitem__(self, word):
        if word in self.specials:
            raise ValueError("can not delete special token")

        if word not in self:
            return

        index = self.word2index[word]
        del self.word2index[word]
        del self.index2word[index]
        del self.freqs[word]
