"""
Bert_Model.py File
Module for running textual analysis
Loads the model and runs inference on the users input
Results are a fake/real probability and a flag
"""

# Dependencies
import torch
import os
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging

# Prevent Console Logs
logging.set_verbosity_error()

# Variable Configuration
bert_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "hamzab-roberta-fake-news-classification")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model Loading
@st.cache_resource
def load_bert():

    # Loading processor and model from local path
    # Debugging Code
    #print("Loading Bert Model...")
    tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(bert_model_path).to(device)
    model.eval()
    # Debugging Code
    #print("Bert Model Loaded.")
    return tokenizer, model

# Inference Function
# Inputs: Text - String
# Outputs: Dictionary with fake probability, real probability and flagged
def run_bert(text):

    # Loading in the model
    tokenizer, model = load_bert()

    # Processing the image and caption
    inputs = tokenizer(text,
                       return_tensors = "pt",
                       truncation = True,
                       max_length = 128,
                       padding = True).to(device)
    
    # Run the inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Calculating mismatch probability (inverse of prediction)
    probs = torch.nn.functional.softmax(outputs.logits, dim = 1)
    fake_prob = round(probs[:, 0].item(), 4)
    real_prob = round(probs[:, 1].item(), 4)
    prediction = torch.argmax(outputs.logits, dim = 1).item()

    return {
        "fake_prob": fake_prob,
        "real_prob": real_prob,
        "prediction": prediction
    }
