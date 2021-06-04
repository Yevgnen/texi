# -*- coding: utf-8 -*-

from __future__ import annotations

import abc
import itertools
import random
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import torch
from carton.collections import collate
from transformers.tokenization_utils import PreTrainedTokenizer

from texi.preprocessing import LabelEncoder
from texi.pytorch.dataset.collator import Collator
from texi.utils import ModeKeys


class PreTrainedCollator(Collator, metaclass=abc.ABCMeta):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        label_encoder: Optional[LabelEncoder] = None,
        mode: ModeKeys = ModeKeys.TRAIN,
    ) -> None:
        super().__init__(mode=mode)
        self.tokenizer = tokenizer
        if label_encoder is None:
            label_encoder = LabelEncoder()
        self.label_encoder = label_encoder

    def collate_fn(self, batch: Sequence) -> Any:
        return self._collate(batch)


class TextClassificationCollator(PreTrainedCollator):
    def collate_train(
        self, batch: Sequence
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        collated = collate(batch)

        x = self.tokenizer(
            collated["text"],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        y = self.label_encoder.encode(collated["label"], return_tensors="pt")

        return x, y


class TextMatchingCollator(PreTrainedCollator):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        label_encoder: Optional[LabelEncoder] = None,
        mode: ModeKeys = ModeKeys.TRAIN,
    ) -> None:
        super().__init__(tokenizer, label_encoder=label_encoder, mode=mode)

    def collate_train(
        self, batch: Sequence
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        # {
        #     "texts": ["sentence1", "sentence2", ...]
        #     "label": 1
        # }

        collated = collate(batch)

        batch_size = len(batch)
        texts = list(itertools.chain.from_iterable(zip(*collated["texts"])))
        assert (
            len(texts) % batch_size == 0
        ), 'All exmaples should have size of "texts" fields'

        x = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        chunks = len(texts) // batch_size

        def _stack(t):
            return torch.stack(t.chunk(chunks, dim=0), dim=0)

        x = {k: _stack(v) for k, v in x.items()}
        y = self.label_encoder.encode(collated["label"], return_tensors="pt")

        return x, y


class MaskedLMCollator(PreTrainedCollator):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        mlm_probability: float = 0.15,
        strict: bool = False,
        ignore_index: int = -100,
        mode: ModeKeys = ModeKeys.TRAIN,
    ) -> None:
        super().__init__(tokenizer=tokenizer, mode=mode)
        self.mlm_probability = torch.tensor(mlm_probability)
        self.strict = strict
        self.ignore_index = ignore_index

    def _whole_word_mask(self, words, tokens):
        # Create MLM mask with `self.mlm_probability`.
        special_token_mask = torch.tensor(
            [token in self.tokenizer.all_special_tokens for token in tokens]
        )
        prob = torch.full((len(tokens),), self.mlm_probability)
        prob.masked_fill_(special_token_mask, 0)
        mask = torch.bernoulli(prob).bool()

        num_tokens = (~special_token_mask).sum()
        num_masked_tokens = int(self.mlm_probability * num_tokens)

        word_iter = iter(words)
        i = 0
        spans = []

        # Loop all token pieces until we have masked enough tokens.
        while i < len(tokens):
            token = tokens[i]

            # Don't mask special tokens.
            if token in self.tokenizer.all_special_tokens:
                i += 1
                continue

            # When we find a new word,
            if not token.startswith("##"):
                word = next(word_iter)

                selected = mask[i]
                j = 1

                # if it is not a single-piece word, we need to find all
                # pieces of it and check if any piece is selected to be
                # masked.
                if [token] != self.tokenizer.basic_tokenizer.tokenize(word):

                    # if the word is split into piece with explicit markers
                    while j < len(word) and tokens[i + j].startswith("##"):
                        selected |= mask[i + j]
                        j += 1

                    # or it is split simply by chars
                    if j == 1:
                        while j < len(word):
                            selected |= mask[i + j]
                            j += 1

                # Mask all piece of current word.
                if selected:
                    mask[i : i + j] = True

                    # Break if we have masked enough tokens.
                    num_masked_tokens -= j

                spans += [(i, i + j)]
                i += j

        assert next(word_iter, None) is None
        assert len(spans) == len(words)

        if self.strict:
            # Select random token piece spans,
            random.shuffle(spans)

            # to reduce masked tokens.
            for start, end in spans:
                mask[start:end] = False
                num_masked_tokens += end - start

                if num_masked_tokens > 0:
                    break

        return mask

    def collate_train(self, batch: Sequence[Mapping]) -> Any:
        collated = collate(batch)

        inputs = self.tokenizer(
            collated,
            is_split_into_words=True,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_id = inputs["input_ids"]
        label = input_id.clone()

        whole_word_mask = torch.stack(
            [
                self._whole_word_mask(
                    words, self.tokenizer.convert_ids_to_tokens(sample_input_ids)
                )
                for words, sample_input_ids in zip(collated, input_id.tolist())
            ],
            dim=0,
        )
        label[~whole_word_mask] = self.ignore_index

        replace_mask = (
            torch.bernoulli(torch.full(input_id.size(), 0.8)).bool() & whole_word_mask
        )
        input_id[replace_mask] = self.tokenizer.mask_token_id

        random_mask = (
            torch.bernoulli(torch.full(input_id.size(), 0.1)).bool()
            & whole_word_mask
            & ~replace_mask
        )
        random_words = torch.randint(len(self.tokenizer), input_id.size())
        input_id[random_mask] = random_words[random_mask]

        inputs["input_ids"] = input_id

        return inputs, label
