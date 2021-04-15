# -*- coding: utf-8 -*-

import torch


def random_entity_mask(num_entities, max_length=20):
    start = torch.randint(0, max_length, size=(num_entities,))
    end = torch.randint(0, max_length, size=(num_entities,))

    start, end = torch.min(start, end), torch.max(start, end)
    mask = torch.arange(max_length).view(max_length, -1)
    mask = (mask >= start) & (mask < end)
    mask = mask.transpose(0, 1)

    return mask
