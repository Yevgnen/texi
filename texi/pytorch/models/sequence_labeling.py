# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel

from texi.pytorch.masking import length_to_mask
from texi.pytorch.rnn import LSTM


class BiLstmCrf(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_labels,
        embedded_size=100,
        hidden_size=100,
        num_layers=2,
        char_embedding=None,
        dropout=0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_labels = num_labels
        if char_embedding is None:
            char_embedding = nn.Embedding(vocab_size, embedded_size)
        self.char_embedding = char_embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.encoder = LSTM(
            embedded_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
        )
        self.fc = nn.Linear(2 * hidden_size, num_labels)

    def forward(self, inputs):
        x = inputs["token"]
        lengths = inputs["length"]

        embedded = self.char_embedding(x)
        embedded = self.embedding_dropout(embedded)
        hidden, _ = self.encoder(embedded, lengths)
        logits = self.fc(hidden)

        return logits


class BertForSequenceLabeling(nn.Module):
    def __init__(self, pretrained_model, **kwargs):
        super().__init__()
        num_labels = kwargs.get("num_labels")
        if num_labels is None:
            raise KeyError("Required key `num_labels` not give.")

        if isinstance(pretrained_model, str):
            self.bert = BertModel.from_pretrained(
                pretrained_model, num_labels=num_labels
            )
        else:
            self.bert = pretrained_model

        self.output = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, inputs):
        inputs = {
            key: value
            for key, value in inputs.items()
            if key not in {"offset_mapping", "tag_mask"}
        }

        outputs = self.bert(**inputs)
        logits = self.output(outputs[0])

        return logits


class SequenceCrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, input_, target, tag_mask):
        assert input_.size()[:-1] == tag_mask.size()
        assert target.size() == tag_mask.size()

        tag_mask = tag_mask.view(-1).bool()
        input_ = input_.view(-1, input_.size()[-1])[tag_mask]
        target = target.view(-1)[tag_mask]

        return super().forward(input_, target)


class CRFForPreTraining(CRF):
    def __init__(self, num_tags, batch_first=True):
        super().__init__(num_tags, batch_first=batch_first)

    def forward(self, emissions, tags, mask, reduction="sum"):
        emissions = emissions[:, 1:, :]
        tags = tags[:, 1:]
        mask = mask[:, 1:].bool()

        return -super().forward(emissions, tags, mask, reduction=reduction)

    def decode(self, emissions, mask):
        return super().decode(emissions, mask.bool())


class CRFDecoder(object):
    def __init__(self, tokenizer, label_encoder, crf):
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.crf = crf

    def decode_tokens(self, x):
        tokens = self.tokenizer.batch_decode(x["token"], x["length"])
        tokens = [x[:length] for x, length in zip(tokens, x["length"])]

        return tokens

    def decode_tags(self, x, y):
        if y.dim() > 2:
            tags = self.crf.decode(y, length_to_mask(x["length"], batch_first=True))
        else:
            tags = y.detach().cpu()
        tags = self.label_encoder.decode(tags)

        return [x[:length] for x, length in zip(tags, x["length"])]

    def decode(self, batch):
        x, y = batch

        tokens = self.decode_tokens(x)
        tags = self.decode_tags(x, y)

        assert len(tokens) == len(tags)
        assert all(
            len(sample_tokens) == len(sample_tags)
            for sample_tokens, sample_tags in zip(tokens, tags)
        )

        return tokens, tags


class SequenceDecoderForPreTraining(object):
    def __init__(self, tokenizer, label_encoder):
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder

    def decode_tokens(self, x, token_transform):
        tokens = []
        for input_ids, offsets in zip(x["input_ids"], x["offset_mapping"]):
            # Decode token ids.
            sample_tokens, token_ids = [], []
            # Loop over all subwords.
            for i, offset in enumerate(offsets):
                # Ignore special tokens.
                if offset.sum() == 0:
                    continue

                # Found all tokens of a word.
                if offset[0] == 0 and token_ids:
                    sample_tokens += [self.tokenizer.convert_ids_to_tokens(token_ids)]
                    token_ids = []

                # Accumulate subwords of current word.
                token_ids += [input_ids[i]]

            if token_ids:
                sample_tokens += [self.tokenizer.convert_ids_to_tokens(token_ids)]
                token_ids = []
            tokens += [[*map(token_transform, sample_tokens)]]

        return tokens

    def decode_tags(self, x, y):
        tags = []
        for tag_mask, tag_ids in zip(x["tag_mask"], y):
            # Decode tag ids.
            if tag_ids.dim() > 1:  # logit
                tag_ids = tag_ids.argmax(dim=-1)
            tag_ids = tag_ids[tag_mask > 0].detach().cpu().numpy().tolist()
            tags += [tag_ids]
        tags = self.label_encoder.decode(tags)

        return tags

    def decode(
        self, batch, token_transform=lambda x: "".join(w.replace("##", "") for w in x)
    ):
        x, y = batch

        tokens = self.decode_tokens(x, token_transform)
        tags = self.decode_tags(x, y)

        assert len(tokens) == len(tags)
        assert all(
            len(sample_tokens) == len(sample_tags)
            for sample_tokens, sample_tags in zip(tokens, tags)
        )

        return tokens, tags


class CRFDecoderForPreTraining(SequenceDecoderForPreTraining):
    def __init__(self, tokenizer, label_encoder, crf):
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.crf = crf

    def decode_tags(self, x, y):
        if y.dim() == 2:
            return super().decode_tags(x, y)

        tags = self.crf.decode(y[:, 1:, :], x["tag_mask"][:, 1:])
        tags = self.label_encoder.decode(tags)

        return tags


class SequenceLabeler(object):
    def __init__(self, net, tokenizer, label_encoder, dataset_class, tagger):
        self.net = net
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.dataset_class = dataset_class
        self.tagger = tagger

    def predict(self, tokens, device="cpu", non_blocking=False, batch_size=32):
        dummy_tag = self.label_encoder.vocab[0]

        examples = [
            {"tokens": sample_tokens, "tags": [dummy_tag] * len(sample_tokens)}
            for sample_tokens in tokens
        ]
        dataset = self.dataset_class(
            examples,
            tokenizer=self.tokenizer,
            label_encoder=self.label_encoder,
            train=False,
        )
        data_loader = dataset.get_dataloader(sampler_kwargs={"batch_size": batch_size})

        outputs = []
        self.net.eval()
        with torch.no_grad():
            for batch in data_loader:
                batch = convert_batch(batch, device=device, non_blocking=non_blocking)
                x = batch[0]
                logit = self.net(x)
                y_pred = logit.argmax(dim=-1)
                outputs += [
                    {"text": tokens, "tag": tags}
                    for tokens, tags in zip(*dataset.decode_seq((x, y_pred)))
                ]
        outputs = [*map(self.tagger.decode, outputs)]

        return outputs
