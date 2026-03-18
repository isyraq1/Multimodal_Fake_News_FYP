### Dependencies Required ###
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification)
import os

# Model Names
model_names = ["mrm8488/bert-tiny-finetuned-fake-news-detection",
               "hamzab/roberta-fake-news-classification",
               "jy46604790/Fake-News-Bert-Detect",
               "elozano/bert-base-cased-fake-news",
               "Arko007/fake-news-roberta-5M"]

# Downloading the model
def download_models(model_name):
    print(f"Downloading Model: {model_name}")

    # Retrieving the BERT models
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Saving the models
    path_loc = os.path.dirname(os.path.abspath(__file__))
    save_dir = (path_loc + "/models")
    safe_name = model_name.replace("/", "-")

    print("Saving Model...")
    model.save_pretrained(save_dir + "/" + safe_name)
    tokenizer.save_pretrained(save_dir + "/" + safe_name)

# Run the model downloads
for model in model_names:
    download_models(model)