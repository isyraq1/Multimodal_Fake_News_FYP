"""
Blip_Model.py File
Module for running visual analysis
Loads the model and runs inference on the users input
Results are inversed to provide a mismatch score
"""

# Dependencies
import os
import torch
import streamlit as st
from PIL import Image 
from transformers import AutoProcessor, BlipForImageTextRetrieval, logging

# Prevent Console Logs
logging.set_verbosity_error()

# Variable Configuration
blip_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "Salesforce-blip-itm-large-flickr")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model Loading
@st.cache_resource
def load_blip():

    # Loading processor and model from local path
    # Debugging Code
    #print("Loading Blip Model...")
    processor = AutoProcessor.from_pretrained(blip_model_path)
    model = BlipForImageTextRetrieval.from_pretrained(blip_model_path).to(device)
    model.eval()
    # Debugging Code
    #print("Blip Model Loaded.")
    return processor, model

# Inference Function
# Inputs: Image - PIL Image and Caption - String
# Outputs: Dictionary with mismatch score and match probability
def run_blip(image, caption):

    # Loading in the model
    processor, model = load_blip()

    # Processing the image and caption
    inputs = processor(images = image,
                       text = caption,
                       return_tensors = "pt",
                       truncation = True,
                       max_length = 128).to(device)
    
    # Run the inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Calculating mismatch probability (inverse of prediction)
    prob = torch.nn.functional.softmax(outputs.itm_score, dim = 1)[:, 1].item()
    match_prob = round(prob, 4)
    mismatch_prob = round(1.0 - match_prob, 4)

    return {
        "mismatch_prob": mismatch_prob,
        "match_prob": match_prob
    }
