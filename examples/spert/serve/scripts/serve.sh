#!/bin/bash

torch-model-archiver \
    --model-name spert \
    --version 1.0 \
    --serialized-file /home/model-server/checkpoint/model.pt \
    --extra-files /home/model-server/checkpoint/params.yaml \
    --extra-files /home/model-server/checkpoint/entity_labels.json \
    --extra-files /home/model-server/checkpoint/relation_labels.json \
    --export-path /home/model-server/checkpoint \
    --handler /home/model-server/scripts/spert_handler.py

torch-model-archiver \
    --model-name spert \
    --version 1.0 \
    --serialized-file checkpoint/model.pt \
    --extra-files checkpoint/params.yaml,checkpoint/entity_labels.json,checkpoint/relation_labels.json,checkpoint/config.json,checkpoint/special_tokens_map.json,checkpoint/vocab.txt \
    --export-path checkpoint \
    --handler /home/user/git/texi/texi/pytorch/plm/spert/serving.py \
    --requirements-file scripts/requirements.txt
