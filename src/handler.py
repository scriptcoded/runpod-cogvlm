""" Example handler file. """

import runpod
from transformers import AutoTokenizer, AutoModel
import urllib.request
from PIL import Image
import requests
from io import BytesIO

# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.

def get_model(model_path):
    """Load a Hugging Face model and tokenizer from the specified directory"""
    model = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    return model, tokenizer

# Load the model and tokenizer
model, tokenizer = get_model('model/')

def handler(job):
    """ Handler function that will be used to process jobs. """
    prompt = job['prompt']
    image_url = job['image']

    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))

    msgs = [{'role': 'user', 'content': prompt}]

    res = model.chat(
        image=img,
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=True, # if sampling=False, beam_search will be used by default
        temperature=0.7,
        # system_prompt='' # pass system_prompt if needed
    )

    return res


runpod.serverless.start({"handler": handler})
