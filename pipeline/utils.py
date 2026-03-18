"""
utils.py file
Contains helper functions for the application
"""

import re

# Text Cleaning Function
# Input:  text (str)
# Output: cleaned text (str)
def text_cleaning(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()