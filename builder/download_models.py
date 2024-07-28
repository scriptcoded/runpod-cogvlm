from transformers import AutoTokenizer, AutoModel
import os

def download_model(model_path, model_name):
    """Download a Hugging Face model and tokenizer to the specified directory"""
    # Check if the directory already exists
    if not os.path.exists(model_path):
        # Create the directory
        os.makedirs(model_path)

    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Save the model and tokenizer to the specified directory
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

download_model('model/', 'sdasd112132/Vision-8B-MiniCPM-2_5-Uncensored-and-Detailed-4bit')
