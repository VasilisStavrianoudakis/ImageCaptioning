import json
import os
from math import ceil
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import GloVe
from torchtext.vocab.vocab import Vocab
from torchvision import transforms


def write_json(filepath: str, data: Any):
    with open(filepath, "w", encoding="utf8") as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False)


def open_json(filepath: str) -> Any:
    with open(filepath, "r", encoding="utf8") as fp:
        data = json.load(fp)
    return data


def get_pretrained_emb(
    vocab: Vocab, embeddings: GloVe, emb_size: int
) -> Tuple[torch.Tensor, List[str]]:
    embs = torch.zeros(len(vocab), emb_size)
    replaced = 0
    not_found = []

    known_words = embeddings.stoi
    vocab_words = vocab.get_stoi()

    for word, idx in vocab_words.items():
        if word in known_words:
            replaced += 1
            embs[idx] = torch.tensor(embeddings[word], dtype=torch.float)
        else:
            not_found.append(word)
            embs[idx] = torch.FloatTensor(emb_size).uniform_(-1.0, 1.0)

    print(
        f"{replaced} words were prefilled out of {len(vocab)} ({round(replaced/len(vocab), 4)})"
    )
    return embs, not_found


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
            transforms.Resize((256, 256)),
            # transforms.Resize((299, 299)),
            transforms.ToTensor(),
            # transforms.Normalize(
            #     (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            # ),  # TODO: REMOVE THIS!
        ]
    )
    image = transform(image)
    image = image.unsqueeze(0)
    # print(image.shape)
    return image


def get_data(
    output_path: str, captions_path: str, images_path: str, test_size: float = 0.1
):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        image_captions = pd.read_csv(captions_path)

        # Pre-process the captions. Convert them to lowercase and and a full stop if it's missing.
        image_captions["caption"] = preprocess_texts(
            captions=image_captions["caption"].tolist()
        )

        # Create the complete path of the images.
        image_captions["image"] = [
            os.path.join(images_path, image)
            for image in image_captions["image"].tolist()
        ]

        # Split the data.
        _, rest = train_test_split(
            image_captions["image"].unique(), test_size=test_size, random_state=1
        )
        validation_set, test_set = train_test_split(rest, test_size=0.5, random_state=1)

        # Choose randomly one caption.
        validation_set = (
            image_captions[image_captions["image"].isin(validation_set)]
            .groupby("image")
            .apply(lambda x: x.sample(1))
            .reset_index(drop=True)
        )

        # Remove the validation image-caption pairs from the training set.
        training_set = (
            pd.merge(image_captions, validation_set, indicator=True, how="outer")
            .query('_merge=="left_only"')
            .drop("_merge", axis=1)
        )

        # Choose randomly one caption.
        test_set = (
            image_captions[image_captions["image"].isin(test_set)]
            .groupby("image")
            .apply(lambda x: x.sample(1))
            .reset_index(drop=True)
        )

        # Remove the test image-caption pairs from the training set.
        training_set = (
            pd.merge(training_set, test_set, indicator=True, how="outer")
            .query('_merge=="left_only"')
            .drop("_merge", axis=1)
        )

        training_set.to_csv(os.path.join(output_path, "train.csv"), index=False)
        validation_set.to_csv(os.path.join(output_path, "validation.csv"), index=False)
        test_set.to_csv(os.path.join(output_path, "test.csv"), index=False)
    else:
        training_set = pd.read_csv(os.path.join(output_path, "train.csv"))
        validation_set = pd.read_csv(os.path.join(output_path, "validation.csv"))
        test_set = pd.read_csv(os.path.join(output_path, "test.csv"))

    return training_set, validation_set, test_set


def freeze_or_unfreeze_layers(model, layers: List[str], freeze: bool = True) -> Any:
    """
    Freezes or unfreezes layers.

    Args:
        `layers` (List[str]):
            Which layers to affect.
        `freeze` (bool, optional):
            `False`: If you want to unfreeze the layers
            `True`: If you want to freeze the layers. Defaults to True.
    """
    for name, param in model.named_parameters():
        print(name, param.requires_grad)

        # If a layer requires_grad and we want to freeze it, freeze variable will be True.
        # So both of them will be True and we will change the requires_grad parameter to not freeze (= False).
        # If requires_grad and we want to unfreeze the layer, freeze will be False.
        # In that case, we won't do anything because the layer requires_grad already.
        if (
            any(layer.lower() in name.lower() for layer in layers)
            and param.requires_grad == freeze
        ):
            param.requires_grad = not freeze
            print(name, param.requires_grad)

    return model


def unfreeze_model(
    model, epoch: int, unfreeze_scheduler: Dict[int, List[str]]
) -> Tuple[Any, Dict[str, int]]:
    events = {}
    for encoder_event, layers in unfreeze_scheduler.items():
        if epoch == int(encoder_event):
            print("\n### UNFREEZING LAYERS ###\n")
            temp = "_".join(layers)
            events[f"unfroze_{temp}"] = epoch

            model = freeze_or_unfreeze_layers(
                model=model,
                layers=layers,
                freeze=False,
            )
    return model, events


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
