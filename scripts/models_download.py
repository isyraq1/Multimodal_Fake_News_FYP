"""
Run this script to download the models required for the application
"""

### Dependencies Required ###
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          AutoProcessor,
                          BlipForImageTextRetrieval)
import os


# Selected Models
bert_model = "hamzab/roberta-fake-news-classification"
blip_model = "Salesforce/blip-itm-large-flickr"

# Downloading the BERT model
def download_bert(model_name):

    print(f"Downloading BERT Model: {model_name}")

    # Retrieving the BERT model
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Saving the model
    path_loc = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = (path_loc + "/models")
    safe_name = model_name.replace("/", "-")

    print(f"Saving Model to {save_dir}/{safe_name}")
    model.save_pretrained(save_dir + "/" + safe_name)
    tokenizer.save_pretrained(save_dir + "/" + safe_name)

# Downloading the BLIP model
def download_blip(model_name):

    print(f"\nDownloading BLip Model: {model_name}")

    # Retrieving the BLIP model
    model = BlipForImageTextRetrieval.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)

    # Saving the model
    path_loc = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = (path_loc + "/models")
    safe_name = model_name.replace("/", "-")

    print(f"Saving Model to {save_dir}/{safe_name}")
    model.save_pretrained(save_dir + "/" + safe_name)
    processor.save_pretrained(save_dir + "/" + safe_name)

print("=" * 30)
print("Downloading Models")

download_bert(bert_model)
download_blip(blip_model)

print("\nModels Downloaded.")