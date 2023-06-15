import json
from math import ceil
from typing import Any, Dict, List, Tuple, Union

import torch
from PIL import Image
from sklearn.utils import shuffle
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab.vocab import Vocab
from torchvision import transforms


def write_json(filepath: str, data: Any):
    with open(filepath, "w", encoding="utf8") as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False)


def open_json(filepath: str) -> Any:
    with open(filepath, "r", encoding="utf8") as fp:
        data = json.load(fp)
    return data


def tokens_generator(texts: List[str]):
    for text in texts:
        yield text.split()


def preprocess_texts(captions: List[str]) -> List[str]:
    captions = [caption.lower() for caption in captions]
    captions = [
        caption if caption[-1] == "." else caption + " ." for caption in captions
    ]
    return captions


def preprocess_image(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose(
        [
            # transforms.Resize((256, 256)),
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            ),  # TODO: REMOVE THIS!
        ]
    )
    image = transform(image)
    image = image.unsqueeze(0)
    # print(image.shape)
    return image


class CustomDataLoader:
    def __init__(
        self,
        images_paths: List[str],
        captions: List[str],
        vocab: Vocab,
        do_shuffle: bool = False,
        batch_size: int = 16,
    ) -> None:
        self.images_paths = images_paths
        # self.captions = captions
        self.vocab = vocab
        self.captions = [
            [vocab[token] for token in caption.split()] for caption in captions
        ]

        self.do_shuffle = do_shuffle
        self.batch_size = batch_size
        self.current_index = 0
        self.length = ceil(len(images_paths) / batch_size)
        self._sanity_checks()

    def _sanity_checks(self):
        assert len(self.images_paths) == len(self.captions)

    def _do_shuffle(self):
        self.images_paths, self.captions = shuffle(self.images_paths, self.captions)
        self._sanity_checks()

    def __len__(self):
        return self.length

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index == len(self.images_paths):
            self.current_index = 0
            # print("End of iteration.")
            if self.do_shuffle:
                self._do_shuffle()
            raise StopIteration

        # find the end of the batch
        end = min(
            self.current_index + self.batch_size,
            len(self.images_paths),
        )

        # get the data for this batch
        imgs = [
            Image.open(img).convert("RGB")
            for img in self.images_paths[self.current_index : end]
        ]
        imgs = [preprocess_image(img) for img in imgs]
        # imgs = [img.unsqueeze(0) for img in imgs]
        imgs = torch.cat(imgs, dim=0)
        # print(imgs.shape)

        true_captions = [caption for caption in self.captions[self.current_index : end]]
        lengths = [len(caption) for caption in true_captions]
        true_captions = [torch.LongTensor(caption) for caption in true_captions]

        # Remove the full stop because we want the model to stop generating after this.
        model_captions = [caption[:-1] for caption in true_captions]
        model_captions = pad_sequence(model_captions, batch_first=True)

        true_captions = pad_sequence(true_captions, batch_first=True)

        self.current_index = end

        return imgs, true_captions, model_captions, lengths
