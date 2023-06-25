import json
import multiprocessing
import os
import queue
import re
from math import ceil
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from num2words import num2words
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import GloVe
from torchtext.vocab.vocab import Vocab
from torchvision import transforms
from tqdm import tqdm


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


def remove_redundant_spaces(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def preprocess_texts(captions: List[str]) -> List[str]:
    captions = [caption.lower() for caption in captions]
    captions = [
        caption.replace("#", "number").replace("&", "and") for caption in captions
    ]
    # Remove everything that it's in parentheses.
    captions = [re.sub(r"\(.+\)", "", caption) for caption in captions]
    captions = [re.sub(r"[\.\;\!:\(\)]", "", caption) for caption in captions]
    # Remove 'a' because it is the most common token.
    captions = [
        re.sub(r"(?:(?<=\W)|(?<=^))a(?:(?=\W)|(?=$))", "", caption)
        for caption in captions
    ]
    captions = [remove_redundant_spaces(caption) for caption in captions]

    # Numbers to words.
    captions = [
        " ".join(
            [
                num2words(token) if token.isdigit() else token
                for token in caption.split()
            ]
        )
        for caption in captions
    ]

    for idx, caption in enumerate(captions):
        if caption:
            captions[idx] = caption if caption[-1] == "." else caption + " ."

    return captions


def get_transformations(will_be_saved: bool) -> List[Any]:
    transformations = [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]
    if will_be_saved:
        transformations.append(transforms.ToPILImage())
    return transformations


def preprocess_image(
    image: Image.Image, will_be_saved: bool = False
) -> Union[torch.Tensor, Image.Image]:
    transformations = get_transformations(will_be_saved=will_be_saved)

    transform = transforms.Compose(transformations)
    image = transform(image)
    if will_be_saved:
        return image

    image = image.unsqueeze(0)
    # print(image.shape)
    return image


def _get_num_parallel_jobs(n_jobs: int) -> int:
    if n_jobs < -1 or n_jobs == 0:
        raise ValueError("Please provide a positive number or -1 as the n_jobs.")
    cores = multiprocessing.cpu_count()
    if n_jobs == -1 or n_jobs > cores:
        return cores
    else:
        return n_jobs


def _use_threads(
    q,
    lock,
    images_path: str,
    new_output_path: str,
):
    # Create a custom progress bar. This progress bar is shared between all processes.
    init_size = q.qsize()
    with lock:
        bar = tqdm(
            desc="Preprocessing and saving images",
            total=init_size,
            leave=False,
        )

    new_training_set = []
    while not q.empty():
        try:
            df = q.get(True, 1.0)
            images_name = df["image"].tolist()
            imgs = [
                Image.open(os.path.join(images_path, img_name)).convert("RGB")
                for img_name in images_name
            ]
            imgs = [preprocess_image(img, will_be_saved=True) for img in imgs]
            new_images_name = [
                img_name.replace(".jpg", f"_{i}.jpg")
                for i, img_name in enumerate(images_name)
            ]
            for new_img_name, img in zip(new_images_name, imgs):
                img.save(os.path.join(new_output_path, new_img_name))

            df["image"] = new_images_name

            new_training_set.append(df)

            # Update the progress bar.
            # A child process sees the remaining texts inside the shared queue and knows that we have processed (init_size - queue_size) texts so far.
            # So it has to update the progress bar by that number.
            # Example: init_size = 10 and the first time that a child process tries to update the progress bar, the queue has 6 texts left.
            # So the first update must "move" the progress bar by (10 - 6) = 4.
            size = q.qsize()
            with lock:
                bar.update(init_size - size)
                init_size = size
        except queue.Empty:
            pass
        except Exception as e:
            print(e)
    # Close the progress bar and then return.
    with lock:
        bar.close()
    return new_training_set


def _create_with_threads(
    q,
    n_jobs: int,
    images_path: str,
    new_output_path: str,
):
    lock = multiprocessing.Manager().Lock()

    pool_args = []
    for _ in range(n_jobs):
        pool_args.append(
            (
                q,
                lock,
                images_path,
                new_output_path,
            )
        )

    with multiprocessing.Pool(n_jobs) as p:
        # print(pool_args)
        results = p.starmap(_use_threads, pool_args)

    # Remove empty lists.
    results = [res for res in results if res]

    new_training_set = []
    for df in results:
        new_training_set.extend(df)
    return new_training_set


def create_data(
    images_path: str,
    new_output_path: str,
    df_set: pd.DataFrame,
) -> pd.DataFrame:
    os.makedirs(new_output_path, exist_ok=True)
    df_set_ = [x for _, x in df_set.groupby("image")]
    del df_set
    m = multiprocessing.Manager()
    q = m.Queue()
    n_jobs = _get_num_parallel_jobs(n_jobs=-1)

    for df in df_set_:
        q.put_nowait(df)

    new_training_set = _create_with_threads(
        q=q, n_jobs=n_jobs, images_path=images_path, new_output_path=new_output_path
    )
    new_training_set = pd.concat(new_training_set)
    return new_training_set


def get_and_create_data(
    output_path: str,
    captions_path: str,
    images_path: str,
    preprocessed_images_output_path: Optional[str] = None,
    test_size: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(output_path):
        image_captions = pd.read_csv(captions_path)

        # Pre-process the captions.
        image_captions["caption"] = preprocess_texts(
            captions=image_captions["caption"].tolist()
        )

        # Drop the data with an empty caption.
        image_captions["caption"].replace("", np.nan, inplace=True)
        image_captions.dropna(subset=["caption"], inplace=True)

        image_captions["lengths"] = [
            len(caption.split()) for caption in image_captions["caption"].tolist()
        ]

        # For each image keep the smallest, in length, caption.
        image_captions = image_captions.loc[
            image_captions.groupby("image")["lengths"].idxmin()
        ]

        # Split the data.
        training_set, rest = train_test_split(
            image_captions, test_size=test_size, random_state=1
        )
        validation_set, test_set = train_test_split(rest, test_size=0.5, random_state=1)
        del rest

        if preprocessed_images_output_path is not None:
            training_set = create_data(
                images_path=images_path,
                new_output_path=preprocessed_images_output_path,
                df_set=training_set,
            )
            validation_set = create_data(
                images_path=images_path,
                new_output_path=preprocessed_images_output_path,
                df_set=validation_set,
            )
            test_set = create_data(
                images_path=images_path,
                new_output_path=preprocessed_images_output_path,
                df_set=test_set,
            )

        os.makedirs(output_path)
        training_set.to_csv(os.path.join(output_path, "train.csv"), index=False)
        validation_set.to_csv(os.path.join(output_path, "validation.csv"), index=False)
        test_set.to_csv(os.path.join(output_path, "test.csv"), index=False)
    else:
        training_set = pd.read_csv(os.path.join(output_path, "train.csv"))
        validation_set = pd.read_csv(os.path.join(output_path, "validation.csv"))
        test_set = pd.read_csv(os.path.join(output_path, "test.csv"))

    if preprocessed_images_output_path is None:
        preprocessed_images_output_path = images_path

    # Create the complete path of the images.
    training_set["image"] = [
        os.path.join(preprocessed_images_output_path, image)
        for image in training_set["image"].tolist()
    ]
    validation_set["image"] = [
        os.path.join(preprocessed_images_output_path, image)
        for image in validation_set["image"].tolist()
    ]
    test_set["image"] = [
        os.path.join(preprocessed_images_output_path, image)
        for image in test_set["image"].tolist()
    ]

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
        start_token: str,
        do_shuffle: bool = False,
        batch_size: int = 16,
    ) -> None:
        self.images_paths = images_paths
        # self.captions = captions
        self.vocab = vocab
        # Add the start token.
        captions = [start_token + " " + caption for caption in captions]
        # print(captions[:2])
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
        lengths = [len(caption) - 1 for caption in true_captions]
        true_captions = [torch.LongTensor(caption) for caption in true_captions]

        # Remove the full stop because we want the model to stop generating after this.
        model_captions = [caption[:-1] for caption in true_captions]
        model_captions = pad_sequence(model_captions, batch_first=True)

        # Remove the start_token because this isn't part of the original caption.
        true_captions = [caption[1:] for caption in true_captions]
        true_captions = pad_sequence(true_captions, batch_first=True)

        self.current_index = end

        return imgs, true_captions, model_captions, lengths
