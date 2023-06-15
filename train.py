import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from torchtext.vocab import build_vocab_from_iterator
from utils import preprocess_texts, tokens_generator

if __name__ == "__main__":
    script_file = Path(__file__).resolve().parent
    data_path = os.path.join(script_file, "data")
    images_paths = os.path.join(data_path, "Images")
    captions_path = os.path.join(data_path, "captions.txt")

    image_captions = pd.read_csv(captions_path)

    # Pre-process the captions. Convert them to lowercase and and a full stop if it's missing.
    image_captions["caption"] = preprocess_texts(
        captions=image_captions["caption"].tolist()
    )
    image_captions["lengths"] = [
        len(caption.split()) for caption in image_captions["caption"].tolist()
    ]
    print(image_captions)

    # For each image keep the smallest in length caption.
    image_captions = image_captions.loc[
        image_captions.groupby("image")["lengths"].idxmin()
    ]
    print(image_captions)

    # Split the data.
    training_set, rest = train_test_split(image_captions, test_size=0.3, random_state=1)
    validation_set, test_set = train_test_split(rest, test_size=0.5, random_state=1)
    del rest, image_captions

    # Build the vocab which includes all tokens with at least min_freq occurrences in the texts.
    # Special tokens <PAD> and <UNK> are used for padding sequences and unknown words respectively.
    vocab = build_vocab_from_iterator(
        tokens_generator(texts=training_set["caption"].tolist()),
        min_freq=5,
        specials=["<PAD>", "<UNK>"],
        special_first=True,
    )
    vocab.set_default_index(vocab["<UNK>"])

    # stoi = vocab.get_stoi()
    # print(stoi)
    # print(max(list(stoi.values())))
    # print(vocab["eeeee"])

    # print(training_set["caption"].tolist()[:2])
    # t = [
    #     [vocab[token] for token in caption.split()]
    #     for caption in training_set["caption"].tolist()
    # ]

    # print(t[:2])

    # images_paths = image_captions["image"]
    # captions = image_captions["caption"]
