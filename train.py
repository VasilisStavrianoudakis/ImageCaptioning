import os
from datetime import datetime
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchinfo import summary
from torchmetrics.functional import bleu_score

# from torchmetrics.functional.text.rouge import rouge_score
from torchtext.vocab import GloVe, build_vocab_from_iterator
from torchtext.vocab.vocab import Vocab
from torchvision import transforms
from tqdm import tqdm

from models import Decoder, Encoder, ImageCaptioningModel
from plotter import plot_info
from utils import (
    CustomDataLoader,
    freeze_or_unfreeze_layers,
    get_and_create_data,
    get_pretrained_emb,
    open_json,
    preprocess_image,
    preprocess_texts,
    tokens_generator,
    unfreeze_model,
    write_json,
)


def get_image_caption(
    model: ImageCaptioningModel, image_path: str, vocab: Vocab, max_length: int = 25
) -> str:
    img = Image.open(image_path).convert("RGB")
    img = preprocess_image(img).to(device)

    model.eval()
    caption = model.caption_image(image=img, vocab=vocab, max_length=max_length)
    return caption


def eval_model(
    model: ImageCaptioningModel,
    dataloader: Iterator,
    vocab: Vocab,
    optimizer,
    loss_function,
    bleu_n_gram: int,
    device: torch.device,
) -> Tuple[float, float]:
    itos = vocab.get_itos()
    pred_sentences = []
    true_sentences = []
    val_losses = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval_Step", leave=False):
            optimizer.zero_grad()  # zero the parameter gradients

            images, true_captions, model_captions, lengths = batch
            images = images.to(device)
            true_captions = true_captions.to(device)
            model_captions = model_captions.to(device)

            outputs = model(images, model_captions, lengths)
            loss = loss_function(
                outputs.reshape(-1, outputs.shape[2]), true_captions.reshape(-1)
            )

            # loss.backward()
            val_losses.append(loss.item())

            # print(outputs.shape)
            model_output = torch.argmax(outputs, dim=2).tolist()
            true_captions = true_captions.tolist()

            pred_sentences_ = [
                [
                    itos[pred_idx]
                    for pred_idx, true_idx in zip(pred_sentence, true_sentence)
                    if true_idx != vocab["<PAD>"]
                ]
                for pred_sentence, true_sentence in zip(model_output, true_captions)
            ]
            true_sentences_ = [
                [
                    itos[true_idx]
                    for true_idx in true_sentence
                    if true_idx != vocab["<PAD>"]
                ]
                for true_sentence in true_captions
            ]

            pred_sentences.extend([" ".join(sentence) for sentence in pred_sentences_])
            true_sentences.extend([" ".join(sentence) for sentence in true_sentences_])

    score = bleu_score(pred_sentences, true_sentences, n_gram=bleu_n_gram).item()
    return score, torch.tensor(val_losses).mean().item()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_path = Path(__file__).resolve().parent
    data_path = os.path.join(script_path, "data")
    output_path = os.path.join(data_path, "csv")
    training_images_output_path = os.path.join(data_path, "training_images")
    images_path = os.path.join(data_path, "Images")
    captions_path = os.path.join(data_path, "captions.txt")

    config = open_json(filepath=os.path.join(script_path, "config.json"))

    inference_path = "inference"
    pretrained = config["pretrained"]
    suffix = "pretrained" if pretrained else "end2end"
    checkpoint_name = "checkpoint.pt"
    vocab_name = "vocab.pt"
    model_path = os.path.join(
        inference_path,
        f"{str(datetime.now().isoformat())}_{suffix}",
    )
    figs_path = os.path.join(model_path, "figs")
    txt_path = os.path.join(model_path, "validation_image_caption.txt")
    model_structure_path = os.path.join(model_path, "model_structure.txt")

    # df = pd.read_csv(captions_path)

    # df["new_caption"] = preprocess_texts(captions=df["caption"].tolist())
    # df.to_csv("seethis.csv", index=False)
    # exit()
    training_set, validation_set, test_set = get_and_create_data(
        output_path=output_path,
        training_images_output_path=training_images_output_path,
        captions_path=captions_path,
        images_path=images_path,
        test_size=0.2,
    )
    # exit()

    # Build the vocab which includes all tokens with at least min_freq occurrences in the texts.
    # Special tokens <PAD> and <UNK> are used for padding sequences and unknown words respectively.

    # from collections import Counter

    # c = Counter()
    # for tokens in tokens_generator(texts=training_set["caption"].tolist()):
    #     c.update(tokens)
    # print(c)
    # print(c.most_common(10))

    # df = pd.DataFrame(data={"token": list(c.keys()), "count": list(c.values())})
    # print(df)

    # print(df[df["count"] <= 10])
    # df = pd.read_csv(captions_path)
    # df["new_caption"] = preprocess_texts(captions=df["caption"].tolist())
    # df.to_csv("seethis.csv", index=False)
    # exit()
    start_token = config["start_token"]
    vocab = build_vocab_from_iterator(
        tokens_generator(texts=training_set["caption"].tolist()),
        min_freq=config["vocab_min_freq"],
        specials=["<PAD>", "<UNK>", start_token],
        special_first=True,
    )
    vocab.set_default_index(vocab["<UNK>"])
    config["decoder_params"]["vocab_size"] = len(vocab)

    training_loader = CustomDataLoader(
        images_paths=training_set["image"].tolist(),
        captions=training_set["caption"].tolist(),
        vocab=vocab,
        do_shuffle=True,
        batch_size=config["batch_size"],
        inference=True,
        start_token=start_token,
    )
    validation_loader = CustomDataLoader(
        images_paths=validation_set["image"].tolist(),
        captions=validation_set["caption"].tolist(),
        vocab=vocab,
        do_shuffle=False,
        batch_size=config["batch_size"],
        inference=True,
        start_token=start_token,
    )
    test_loader = CustomDataLoader(
        images_paths=test_set["image"].tolist(),
        captions=test_set["caption"].tolist(),
        vocab=vocab,
        do_shuffle=False,
        batch_size=config["batch_size"],
        inference=True,
        start_token=start_token,
    )

    # Init the models, optimizer and loss function
    encoder = Encoder(
        lstm_hidden_size=config["decoder_params"]["lstm_hidden_size"],
        **config["encoder_params"],
    ).to(device)
    decoder = Decoder(**config["decoder_params"]).to(device)
    if pretrained:
        embed_size = config["decoder_params"]["embed_size"]
        embeddings = GloVe(name="6B", dim=embed_size)
        vectors, not_found = get_pretrained_emb(
            vocab=vocab, embeddings=embeddings, emb_size=embed_size
        )
        vectors = vectors.to(device)

        # Add the pretrained embeddings.
        decoder.EMBEDDING_LAYER = torch.nn.Embedding.from_pretrained(
            vectors, freeze=config["freeze"], padding_idx=vocab["<PAD>"]
        )

    model = ImageCaptioningModel(
        encoder=encoder, decoder=decoder, start_token=start_token, device=device
    ).to(device)

    loss_function = torch.nn.CrossEntropyLoss(ignore_index=vocab["<PAD>"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    model = freeze_or_unfreeze_layers(
        model=model, layers=["encoder", "decoder"], freeze=config["freeze"]
    )
    print(model)

    stats_per_epoch = {}
    stats_per_epoch["training_loss"] = []
    stats_per_epoch["training_score"] = []
    stats_per_epoch["validation_loss"] = []
    stats_per_epoch["validation_score"] = []
    num_epochs_without_improvement = 0
    patience = config["patience"]
    min_val_loss = np.inf
    max_val_score = np.NINF
    events = {}

    random_val_image = validation_set.sample(1)
    write = []
    write.append(f"Image name: {random_val_image.iloc[0]['image']}\n")
    write.append(f"True caption: {random_val_image.iloc[0]['caption']}\n\n")
    random_val_image_caption = get_image_caption(
        model=model,
        image_path=random_val_image.iloc[0]["image"],
        vocab=vocab,
        max_length=25,
    )
    print("Random val image:", random_val_image.iloc[0]["image"])
    write.append(f"Caption before training: {random_val_image_caption}\n\n")
    # print(random_val_image_caption)
    # exit()

    print("Training in:", device)
    # print(vocab.get_stoi())
    for epoch in tqdm(range(config["epochs"]), desc="Epoch"):
        if config["unfreeze_scheduler"] is not None:
            model, new_events = unfreeze_model(
                model=model,
                epoch=epoch,
                unfreeze_scheduler=config["unfreeze_scheduler"],
            )
            events = {**events, **new_events}
        model.train()
        for batch in tqdm(training_loader, desc="Training", leave=False):
            optimizer.zero_grad()

            images, true_captions, model_captions, lengths = batch
            images = images.to(device)
            true_captions = true_captions.to(device)
            model_captions = model_captions.to(device)

            outputs = model(images, model_captions, lengths)
            loss = loss_function(
                outputs.reshape(-1, outputs.shape[2]), true_captions.reshape(-1)
            )

            loss.backward()
            optimizer.step()

        training_score, training_loss = eval_model(
            model=model,
            dataloader=training_loader,
            vocab=vocab,
            optimizer=optimizer,
            loss_function=loss_function,
            device=device,
            bleu_n_gram=config["bleu_n_gram"],
        )

        validation_score, validation_loss = eval_model(
            model=model,
            dataloader=validation_loader,
            vocab=vocab,
            optimizer=optimizer,
            loss_function=loss_function,
            device=device,
            bleu_n_gram=config["bleu_n_gram"],
        )
        stats_per_epoch["training_loss"].append(training_loss)
        stats_per_epoch["training_score"].append(training_score)
        stats_per_epoch["validation_loss"].append(validation_loss)
        stats_per_epoch["validation_score"].append(validation_score)
        print(
            f"Epoch {epoch}, mean training loss {training_loss:.2f}, mean val loss {validation_loss:.2f}, training bleu: {round(training_score, 3)}, validation blue: {round(validation_score, 3)}"
        )

        random_val_image_caption = get_image_caption(
            model=model,
            image_path=random_val_image.iloc[0]["image"],
            vocab=vocab,
            max_length=25,
        )
        write.append(f"Epoch: {str(epoch + 1)} caption: {random_val_image_caption}\n")
        print("Caption:", random_val_image_caption)
        if validation_loss < min_val_loss:
            num_epochs_without_improvement = 0
            events["best_model"] = epoch
            print(
                f" *** Saving best model. Best validation loss: {min_val_loss} -> {validation_loss}"
            )
            min_val_loss = validation_loss
            os.makedirs(model_path, exist_ok=True)
            # Save the training config.
            if not os.path.exists(os.path.join(model_path, "training_config.json")):
                write_json(
                    filepath=os.path.join(model_path, "training_config.json"),
                    data=config,
                )

            # Save the model.
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    # "optimizer_state_dict": optimizer.state_dict(),
                    "training_loss": training_loss,
                    "training_score": training_score,
                    "validation_loss": validation_loss,
                    "validation_score": validation_score,
                },
                os.path.join(model_path, checkpoint_name),
            )

            # Save the vocab object.
            if not os.path.exists(os.path.join(model_path, vocab_name)):
                torch.save(vocab, os.path.join(model_path, vocab_name))

            # Save model's structure.
            if not os.path.exists(model_structure_path):
                model_stats = summary(
                    model=model,
                    input_data=[images, model_captions, lengths],
                    batch_dim=config["batch_size"],
                    verbose=0,
                )
                summary_str = str(model_stats)
                with open(model_structure_path, "w", encoding="utf8") as fp:
                    fp.write(summary_str)

        else:
            num_epochs_without_improvement += 1

        if patience > 0 and num_epochs_without_improvement >= patience:
            print(
                f"##### Stopping the training phase because there was not any improvement in the last {patience} epochs."
            )
            break

    torch.save(
        {
            "epoch": config["epochs"],
            "model_state_dict": model.state_dict(),
        },
        os.path.join(model_path, f"epochs_{str(config['epochs'])}.pt"),
    )

    os.makedirs(figs_path, exist_ok=True)
    # PLOT TRAINING INFO
    plot_info(
        extra_epochs=-1,
        training_val=stats_per_epoch["training_score"],
        validation_val=stats_per_epoch["validation_score"],
        y_name="BLEU SCORE",
        saveto=os.path.join(figs_path, "model_bleu.png"),
        events=events,
    )
    plot_info(
        extra_epochs=-1,
        training_val=stats_per_epoch["training_loss"],
        validation_val=stats_per_epoch["validation_loss"],
        y_name="LOSS",
        saveto=os.path.join(figs_path, "model_loss.png"),
        events=events,
    )

    # Make sure that the best model is being loaded.
    del model
    checkpoint = torch.load(os.path.join(model_path, checkpoint_name))

    model = ImageCaptioningModel(
        encoder=encoder, decoder=decoder, start_token=start_token, device=device
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    # This is to make sure that the vocab object can be loaded successfully.
    del vocab
    vocab = torch.load(os.path.join(model_path, vocab_name))

    test_score, test_loss = eval_model(
        model=model,
        dataloader=test_loader,
        vocab=vocab,
        optimizer=optimizer,
        loss_function=loss_function,
        device=device,
        bleu_n_gram=config["bleu_n_gram"],
    )

    test_results = (
        f"\n\nTest loss: {round(test_loss, 3)}, Test bleu: {round(test_score, 3)}"
    )
    print(test_results)
    write.append(test_results)
    with open(txt_path, "w", encoding="utf8") as fp:
        fp.writelines(write)

    random_training_image_caption = get_image_caption(
        model=model,
        image_path=training_set.iloc[0]["image"],
        vocab=vocab,
        max_length=25,
    )
    print("Random training image:", training_set.iloc[0]["image"])
    print("Caption:", random_training_image_caption)
