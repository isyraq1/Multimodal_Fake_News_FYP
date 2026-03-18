#  Model Evaluations

This folder contains all evaluation scripts, datasets and results for the 
comparative model selection phase of the project. Evaluations were conducted 
for three pipeline components of BLIP visual analysis, BERT/RoBERTa textual analysis 
and LLM report generation.

## Folder Structure
```
Model Evaluations/
├ BLIP_Model_Evaluation/       BLIP evaluation notebook and results
│   ├ blip_eval_results
├ BERT_Model_Evaluation/       BERT/RoBERTa evaluation notebook and results
│   ├ bert_eval_results
├ fakeddit_images/             Downloaded Fakeddit images for BLIP evaluation
├ LLM_Model_Evaluation/        LLM evaluation script and results
│   ├ llm_evaluation.py
│   ├ fakenews.jpg             Test image used for evaluation
│   └ llm_eval_results/        Generated LLM responses
└ readme.md
```
---

## Datasets

Datasets used are not included in the file repository due to their file sizes.
However, the pre-processed, cleaned and downsized datasets are provided for evaluation running.

If you would like to run the pre-processing notebook, you may find the datasets here:
1. Fakeddit (download the test set ONLY): https://drive.google.com/drive/folders/1DuH0YaEox08ZwzZDpRMOaFpMCeRyxiEF 
2. ISOT: https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/

Please place all downloaded datasets in this directory. 

---

## Model Downloads

5 models were evaluated for BLIP and BERT respectively.
3 models were evaluated for LLM Report Generation.

Due to model size, none of the models are uploaded to github.
To download the models, run their respective model_download_MODEL.py 
found in their respective directories for BLIP and BERT.

### BLIP Candidates
Models:
1. Salesforce/blip-itm-large-flickr (Selected)
2. Salesforce/blip-itm-base-coco
3. Salesforce/blip-itm-large-coco
4. Salesforce/blip-itm-base-flickr
5. Salesforce/blip2-itm-vit-g-coco

### BERT/RoBERTa Candidates
Models:
1. hamzab/roberta-fake-news-classification (Selected)
2. jy46604790/Fake-News-Bert-Detect
3. elozano/bert-base-cased-fake-news
4. Arko007/fake-news-roberta-5M
5. mrm8488/bert-tiny-finetuned-fake-news-detection

### LLM Candidates
Models:
Mistral Large (Score: 13/15 - Selected)
Gemini 2.5 Flash (Score: 13/15)
Groq Llama 3 70B (Score: 9/15)

---

## Running LLM Evaluation

The LLM evaluation script requires API keys for all three candidate models.
In addition to the Mistral AI API Key for .env, please retrieve and add the other API keys
if you with to run model evaluations.

Example final .env:
GEMINI_API_KEY=your_key_here
MISTRAL_API_KEY=your_key_here
GROQ_API_KEY=your_key_here

To obtain API keys:
1. Gemini: https://aistudio.google.com
2. Mistral: https://mistral.ai
3. Llama: https://console.groq.com

**Please run the LLM evaluation from the root directory so that the scripts can retrieve the API keys**