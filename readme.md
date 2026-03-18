# Multimodal Fake News Detection and Education System
# CM3070 Final Project for UOL GOLDSMITH

A web-based application that unifies pre-trained AI models to detect and educate users about fake news.

## Pipeline Breakdown
1. BLIP: Visual Analysis Model for Semantic Analysis
2. RoBERTa: Textual Analysis Model for Textual Fake News
3. DuckDuckGo Search: Search Module with Crafted Queries
4. Mistral Large AI: LLM for Report Generation

---

## Main Project Structure
├ app.py                        Main Streamlit application
├ .env                          API keys (included in .gitignore)
├ requirements.txt              Python dependencies
├ pipeline/
│   ├ blip_model.py             BLIP visual analysis module
│   ├ bert_model.py             RoBERTa textual analysis module
│   ├ search.py                 DuckDuckGo search module
│   ├ llm.py                    Mistral Large report generation module
│   └ utils.py                  Shared utility functions: Includes text_cleaning module only
├ models/                       BLIP & RoBERTa models (models included in .gitignore)
├ tests/
│   ├ unit_tests.py             Unit tests
│   └ integration_tests.py      Integration
├ scripts/
│   └ model_download.py         Script to download required models
└ Model Evaluations/

**Please Refer to the Readme.md File in Model Evaluations for Evaluations**

---

## Application Setup & Run ##
### 1. Install dependencies

Run "pip install -r requirements.txt"

**Note: To use cuda PyTorch must be separately installed from https://pytorch.org based on support for your device**
**else, the application can run on CPU as well. To use without CUDA, install pytorch through pip install**

### 2. Create and add Mistral API Key

1. Sign up and retrieve an api key from https://mistral.ai
2. Create the .env file in this directory
3. Add your Mistral API key as per follows in the .env file: MISTRAL_API_KEY=API_KEY_HERE

### 3. Download the necessary BLIP & RoBERTa Models

Run "python scripts/model_download.py"

### 4. Run the application

Run "streamlit run app.py"
 