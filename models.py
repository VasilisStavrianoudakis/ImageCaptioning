import math
from typing import Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchvision.models as models
from torchtext.vocab.vocab import Vocab


class Encoder(nn.Module):
    def __init__(
        self,
        img_size: int = 256,
        in_channels: int = 3,
        embed_size: int = -1,
        dropout_prob: float = 0.5,
    ):
        super().__init__()
        output_calc = lambda i, k, p, s: int(math.floor(((i + 2 * p - k) / s) + 1))
        out_channels = 16
        m = 2
        k = 7
        s = 1
        p = 1
        self.CONV1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=k,
            stride=s,
            padding=p,
        )
        out_dim = output_calc(i=img_size, k=k, p=p, s=s)
        k = 5
        s = 2
        p = 1
        self.max_pool1 = nn.MaxPool2d(kernel_size=k, stride=s, padding=p)
        out_dim = output_calc(i=out_dim, k=k, p=p, s=s)

        in_channels = out_channels
        out_channels = m * out_channels
        k = 5
        s = 1
        p = 1
        self.CONV2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=k,
            stride=s,
            padding=p,
        )
        out_dim = output_calc(i=out_dim, k=k, p=p, s=s)
        k = 5
        s = 2
        p = 1
        self.max_pool2 = nn.MaxPool2d(kernel_size=k, stride=s, padding=p)
        out_dim = output_calc(i=out_dim, k=k, p=p, s=s)

        in_channels = out_channels
        out_channels = m * out_channels
        k = 3
        s = 1
        p = 1
        self.CONV3 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=k,
            stride=s,
            padding=p,
        )
        out_dim = output_calc(i=out_dim, k=k, p=p, s=s)
        k = 5
        s = 2
        p = 1
        self.max_pool3 = nn.MaxPool2d(kernel_size=k, stride=s, padding=p)
        out_dim = output_calc(i=out_dim, k=k, p=p, s=s)

        self.embed_size = embed_size
        if embed_size > 0:
            # Each matrix in each channel is out_dim x out_dim and we have out_channels number of channels.
            fc_in = out_channels * out_dim * out_dim
            self.FC = nn.Linear(fc_in, embed_size)

        # self.inception = models.inception_v3(pretrained=True, aux_logits=True)
        # self.inception.fc = nn.Linear(self.inception.fc.in_features, 1600)

        self.activation_function = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, images):
        x = self.activation_function(self.CONV1(images))
        x = self.max_pool1(x)
        x = self.activation_function(self.CONV2(x))
        x = self.max_pool2(x)
        x = self.activation_function(self.CONV3(x))
        x = self.max_pool3(x)
        x = torch.flatten(x, start_dim=1)
        if self.embed_size > 0:
            x = self.activation_function(self.FC(x))

        # x = self.inception(images)
        # try:
        #     x = x.logits
        # except AttributeError:
        #     pass
        # x = self.activation_function(x)

        # print(x.shape)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        embed_size: int,
        hidden_size: int,
        vocab_size: int,
        num_layers: int,
        dropout_prob: float = 0.5,
    ):
        super().__init__()
        self.EMBEDDING_LAYER = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_size
        )
        self.LSTM_LAYER = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.FC = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_size, int(hidden_size * 0.5)),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(int(hidden_size * 0.5), vocab_size),
        )

        # self.FC = nn.Linear(hidden_size, vocab_size)
        # self.dropout = nn.Dropout(p=dropout_prob)

    def forward(
        self, features: torch.Tensor, captions: torch.LongTensor, lengths: List[int]
    ):
        # Input: batch_size x seq_length
        x = self.EMBEDDING_LAYER(captions)
        # Output: batch x seq_length x embedding_dimension

        # Add the image context vector as the first element of the sequence.
        x = torch.cat((features.unsqueeze(1), x), dim=1)

        lengths = torch.tensor(lengths, dtype=torch.int16).cpu()

        # pack sequence
        x = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        # Output: batch_sum_seq_len x embedding_dimension
        # Say batch = 2, len(seq1) = 4 and len(seq2) = 3. So batch_sum_seq_len = 4 + 3 = 7

        x, _ = self.LSTM_LAYER(x)

        # Unpack the output.
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        x = self.FC(x)
        return x


class ImageCaptioningModel(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, captions, lengths: list) -> torch.Tensor:
        features = self.encoder(images)
        outputs = self.decoder(features, captions, lengths)
        return outputs

    def caption_image(self, image, vocab: Vocab, max_length=50) -> str:
        itos = vocab.get_itos()
        result_caption = []

        with torch.no_grad():
            x = self.encoder(image).unsqueeze(0)
            # print(x.shape)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoder.LSTM_LAYER(x, states)
                # print(hiddens.shape)
                # print(hiddens.squeeze(0).shape)
                output = self.decoder.FC(hiddens.squeeze(0))
                predicted = output.argmax(1)
                result_caption.append(predicted.item())

                if itos[predicted.item()] == ".":
                    break

                # print(predicted.shape)
                x = self.decoder.EMBEDDING_LAYER(predicted).unsqueeze(0)
                # print(x.shape)
                # print()

        caption = [itos[idx] for idx in result_caption]
        if caption[-1] != ".":
            caption[-1] = "."
        caption[0] = caption[0].capitalize()
        caption = " ".join(caption)
        return caption
