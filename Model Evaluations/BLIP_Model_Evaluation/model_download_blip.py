### Dependencies Required ###
from transformers import (BlipProcessor, 
                          BlipForImageTextRetrieval,
                          AutoProcessor, 
                          Blip2ForImageTextRetrieval)
import os

# Model Names
model_names = ["Salesforce/blip-itm-base-coco",
               "Salesforce/blip-itm-large-coco",
               "Salesforce/blip-itm-base-flickr",
               "Salesforce/blip-itm-large-flickr",
               "Salesforce/blip2-itm-vit-g-coco"]

# Downloading the model
def download_models(model_name):
    print(f"Downloading Model: {model_name}")

    # Blip2 and Blip1 models have different download processes
    if "blip2" in model_name:
        model = Blip2ForImageTextRetrieval.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name)
    else:
        model = BlipForImageTextRetrieval.from_pretrained(model_name)
        processor = BlipProcessor.from_pretrained(model_name)

    # Saving the models
    path_loc = os.path.dirname(os.path.abspath(__file__))
    save_dir = (path_loc + "/models")
    safe_name = model_name.replace("/", "-")

    print("Saving Model...")
    model.save_pretrained(save_dir + "/" + safe_name)
    processor.save_pretrained(save_dir + "/" + safe_name)

# Run the model downloads
for model in model_names:
    download_models(model)