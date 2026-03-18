"""
app.py File
The main application file for the project
Run "streamlit run app.py"
"""

# Dependencies
import os
import re
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from pipeline.blip_model import run_blip
from pipeline.bert_model import run_bert
from pipeline.search import search_news
from pipeline.llm import generate_report
from pipeline.utils import text_cleaning

# Environment Setup (API)
load_dotenv()

### Page Configuration ###
st.set_page_config(
    page_title = "Fake News Detector",
    page_icon = "📰",
    layout = "centered"
)

### Sidebar for Settings & Information ###
with st.sidebar:
    
    st.title("📋 Instructions")
    st.markdown("""
    **Step 1 — Upload Image**
    
    Upload the image that accompanies the article you want to verify.
                
    **Step 2 — Enter Image Caption**
    
    Enter the caption or description accompanying the image in the article.
                
    **Step 3 — Enter Article Headline**
    
    Enter the full headline of the article.
                
    **Step 4 — Enter Article Text**
    
    Paste the full text of the article you want to verify.
                
    **Step 5 — Analyse**
    
    Click the Analyse Article button to run the full pipeline.
                
    **Understanding Results**
    - 📊 **Scores Only** — View Raw Scores Only
    - 📑 **Full Report** — View The Full Report (Default)
    - 👨🏻‍🎓 **Educational Takeaway** — View Catered Media Tips
                
    **Note:** Processing may take up to 3 minutes depending on article length.
    """)

    st.divider()
    st.caption("Multimodal Fake News Detection & Education System")
    st.caption("CM3070 Final Project")

### Page Header ###
st.title("🔍 Multimodal Fake News Detector")
st.write(
    "Submit a news article with its image and accompanying information for a credibility report."
)

st.divider()

### Input Fields ###
st.subheader("Submit Article")

# Image Input
uploaded_image = st.file_uploader(
    "Upload Article Image",
    type = ["jpg", "jpeg", "png"]
)

# Image Caption
image_caption = st.text_input(
    "Image Caption",
    placeholder = "Enter the caption or description of the image"
)

# Article Headline
article_headline = st.text_input(
    "Article Headline",
    placeholder = "Enter the article headline"
)

# Article Text
article_text = st.text_area(
    "Article Text",
    placeholder = "Paste the full article text here",
    height = 200
)

# Submit Button
submit = st.button("Analyse Article", type = "primary")

### Validation & Processing Pipeline ###
if submit:
    
    # Validating inputs are filled
    if not uploaded_image:
        st.warning("Please upload an image before submitting.")
    elif not image_caption.strip():
        st.warning("Please enter an image caption before submitting.")
    elif not article_headline.strip():
        st.warning("Please enter an article headline before submitting.")
    elif not article_text.strip():
        st.warning("Please enter the article text before submitting.")
    else:

        # Processing Pipeline

        # Reset results on new submission
        st.session_state.report = None
        st.session_state.blip_results = None
        st.session_state.bert_results = None
        st.session_state.search_results = None

        # Load and convert image to RGB
        image = Image.open(uploaded_image).convert("RGB")

        # Clean text
        cleaned_text = text_cleaning(article_text)

        # Executing BERT and BLIP Concurrently
        with st.spinner("Analysing image and text..."):
            with ThreadPoolExecutor() as executor:
                blip_future = executor.submit(run_blip, image, image_caption)
                bert_future = executor.submit(run_bert, cleaned_text)
                blip_results = blip_future.result()
                bert_results = bert_future.result()

        # Executing DuckDuckGoSeaarch
        with st.spinner("Searching for relevant sources..."):
            search_results = search_news(article_headline, bert_results, blip_results)

        # Generating Report
        with st.spinner("Generating report..."):
            report = generate_report(
                article_headline,
                image_caption,
                cleaned_text,
                blip_results,
                bert_results,
                search_results
            )

        # Storing Results in Session
        st.session_state.report = report
        st.session_state.blip_results = blip_results
        st.session_state.bert_results = bert_results
        st.session_state.search_results = search_results


### Results Section ###
if "report" in st.session_state and st.session_state.report:
    st.divider()
    st.subheader("Analysis Results")

    # Result buttons
    view = st.radio(
        "Select View",
        ["📑 Full Report", "📊 Scores Only", "👨🏻‍🎓 Educational Takeaway"],
        horizontal = True
    )

    # Full Report Tab
    if view == "📑 Full Report":
        st.markdown(st.session_state.report)

    # Scores Only Tab
    elif view == "📊 Scores Only":
        st.markdown("#### 🖼️ Visual Analysis — BLIP")
        col1, col2 = st.columns(2)
        
        # Providing Match and Mistmatch Probability (Users will not have to calculate)
        with col1:
            st.metric(
                label = "Match Probability",
                value = f"{st.session_state.blip_results['match_prob']:.2%}"
            )

        with col2:
            st.metric(
                label = "Mismatch Probability",
                value = f"{st.session_state.blip_results['mismatch_prob']:.2%}"
            )

        st.divider()

        # Providing Fake Real Probability along with Final Prediction
        st.markdown("#### 📝 Textual Analysis — BERT (RoBERTa)")
        col3, col4, col5 = st.columns(3)
        
        with col3:
            st.metric(
                label = "Fake Probability",
                value = f"{st.session_state.bert_results['fake_prob']:.2%}"
            )
        with col4:
            st.metric(
                label = "Real Probability",
                value = f"{st.session_state.bert_results['real_prob']:.2%}"
            )

        with col5:
            prediction = st.session_state.bert_results['prediction']
            st.metric(
                label = "Prediction",
                value = "FAKE" if prediction == 0 else "REAL"
            )

    # Providing only the educational takeaway
    elif view == "👨🏻‍🎓 Educational Takeaway":
        report_text = st.session_state.report
        # Extract educational takeaway section from report
        match = re.search(
            r'5\.\s*EDUCATIONAL TAKEAWAY\**[:\s]*(.*)',
            report_text,
            re.DOTALL | re.IGNORECASE
        )

        if match:
            extracted = match.group(1).strip()
            # Remove markdown bold markers
            extracted = re.sub(r'\*\*', '', extracted)
            
            st.markdown("#### 👨🏻‍🎓 Educational Takeaway")
            st.markdown(extracted)
            st.markdown("Please refer to the full report for full context.")
        else:
            st.markdown("Please refer to the full report for the educational takeaway.")