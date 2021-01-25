# -*- coding: utf-8 -*-

import torch
from felis.collections import collate

from texi.pytorch.dataset import SequenceLabelingDataset as _SequenceLabelingDataset


class SequenceLabelingDataset(_SequenceLabelingDataset):
    def __init__(self, *args, **kwargs):
        self.dummy_tag = kwargs.pop("dummy_tag", "O")
        super().__init__(*args, **kwargs)

    def collate(self, batch):
        collated = collate(batch)

        x = self.tokenizer(
            collated["tokens"],
            padding=True,
            truncation=True,
            is_split_into_words=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        # Add dummy tags for wordpiece tokens.
        aligned_tags, tag_masks = [], []
        for offsets, tags in zip(x["offset_mapping"], collated["tags"]):
            tags = iter(tags)
            sample_aligned_tags, sample_tag_masks = [], []

            # Loop over all subwords.
            for offset in offsets:

                # Special tokens and subwords other than the first one
                # is labeld as `self.dummy_tag`.
                if offset.sum() == 0 or offset[0] > 0:
                    sample_aligned_tags += [self.dummy_tag]
                    sample_tag_masks += [0]

                # The first token of each word is labeld as its tag.
                else:
                    sample_aligned_tags += [next(tags)]
                    sample_tag_masks += [1]

            aligned_tags += [sample_aligned_tags]
            tag_masks += [sample_tag_masks]

        x["tag_mask"] = torch.tensor(tag_masks, dtype=torch.int64)
        y = self.label_encoder.encode(aligned_tags)

        assert y.size() == x["input_ids"].size()

        return x, y
