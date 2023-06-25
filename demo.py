import os
from pathlib import Path

import gradio as gr
import torch
from models import Decoder, Encoder, ImageCaptioningModel
from utils import open_json, preprocess_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
script_path = Path(__file__).resolve().parent
model_path = os.path.join(script_path, "inference", "best_model")
model_name = "model.pt"

config = open_json(filepath=os.path.join(model_path, "training_config.json"))
vocab = torch.load(os.path.join(model_path, "vocab.pt"))

# Load the models.
checkpoint = torch.load(os.path.join(model_path, model_name), map_location=device)

encoder = Encoder(
    lstm_hidden_size=config["decoder_params"]["lstm_hidden_size"],
    **config["encoder_params"],
).to(device)
decoder = Decoder(**config["decoder_params"]).to(device)

model = ImageCaptioningModel(
    encoder=encoder, decoder=decoder, start_token=config["start_token"], device=device
).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def predict(img):
    img = preprocess_image(image=img)
    caption = model.caption_image(image=img, vocab=vocab, max_length=20)
    return caption


demo = gr.Interface(
    fn=predict,
    inputs=gr.components.Image(label="Image", type="pil"),
    outputs=gr.components.Textbox(label="Image Caption"),
)

demo.launch(inbrowser=True)
