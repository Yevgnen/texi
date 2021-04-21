# -*- coding: utf-8 -*-

import torch
from torch import Tensor
from transformers.file_utils import ModelOutput


def mean_pooling(model_output: ModelOutput, attention_mask: Tensor) -> Tensor:
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(dim=-1).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

    return sum_embeddings / sum_mask


def max_pooling(model_output: ModelOutput, attention_mask: Tensor) -> Tensor:
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(dim=-1).expand(token_embeddings.size()).float()
    )
    token_embeddings[input_mask_expanded == 0] = -1e9
    max_over_time = torch.max(token_embeddings, dim=1)[0]

    return max_over_time


def cls_pooling(model_output: ModelOutput, attention_mask: Tensor) -> Tensor:
    return model_output[0][:, 0]
