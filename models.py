import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 3, p: float = 0.5):
        super().__init__()
        out_channels = 8
        m = 1
        self.CONV1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=7,
            stride=1,
            padding=1,
        )
        self.max_pool1 = nn.MaxPool2d(kernel_size=5, stride=2, padding=0)

        m *= 2
        self.CONV2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels * m,
            kernel_size=5,
            stride=1,
            padding=1,
        )
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        m *= 2
        self.CONV3 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels * m,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.activation_function = nn.ReLU()
        self.dropout = nn.Dropout(p=p)

    def forward(self, images):
        x = self.activation_function(self.CONV1(images))
        x = self.max_pool1(x)
        x = self.activation_function(self.CONV2(x))
        x = self.max_pool2(x)
        x = self.activation_function(self.CONV3(x))
        x = self.max_pool3(x)
        x = torch.flatten(x, start_dim=1)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        embed_size: int,
        hidden_size: int,
        vocab_size: int,
        num_layers: int,
        p: float = 0.5,
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
        self.FC = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(p=p)

    def forward(self, features: torch.Tensor, captions, lengths: list):
        # Input: batch_size x seq_length
        x = self.EMBEDDING_LAYER(captions)
        # Output: batch x seq_length x embedding_dimension

        # Add the image context vector as the first element of the sequence.
        x = torch.cat((features.unsqueeze(0), x), dim=0)
        lengths = [l_ + 1 for l_ in lengths]

        lengths = torch.tensor(lengths, dtype=torch.int16).cpu()

        # pack sequence
        x = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        # Output: batch_sum_seq_len x embedding_dimension
        # Say batch = 2, len(seq1) = 4 and len(seq2) = 3. So batch_sum_seq_len = 4 + 3 = 7

        x, _ = self.LSTM_LAYER(x)
        x = self.FC(x)
        return x


class ImageCaptioningModel(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, captions, lengths: list):
        features = self.encoder(images)
        outputs = self.decoder(features, captions, lengths)
        return outputs

    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []

        with torch.no_grad():
            x = self.encoder(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoder.LSTM_LAYER(x, states)
                output = self.decoder.FC(hiddens.squeeze(0))
                predicted = output.argmax(1)
                result_caption.append(predicted.item())

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

                x = self.decoder.EMBEDDING_LAYER(predicted).unsqueeze(0)

        return [vocabulary.itos[idx] for idx in result_caption]
