from transformers import AutoTokenizer, AutoModel
import os

def download_model(model_path, model_name):
    print("Downloading {model_name} to {model_path}")
    """Download a Hugging Face model and tokenizer to the specified directory"""
    # Check if the directory already exists
    if not os.path.exists(model_path):
        # Create the directory
        os.makedirs(model_path)

    print("Downloading model...")
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Save the model and tokenizer to the specified directory
    print("Saving model...")
    model.save_pretrained(model_path)
    print("Saving tokenizer...")
    tokenizer.save_pretrained(model_path)

    print("Download finished.")

download_model('model/', 'sdasd112132/Vision-8B-MiniCPM-2_5-Uncensored-and-Detailed-4bit')
