"""
llm_evaluation.py
This script utilizes the pipelines in the application
to carry out the evaluation of 3 different LLMs.
The test case used is a fake news source:
http://rightwingnews.com/top-news/famous-dog-killed-spot-waited-year-owner-return/
"""

# Dependencies
import os
import sys
import json
from PIL import Image
from dotenv import load_dotenv

# Ensuring pipeline files can be used
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
load_dotenv(os.path.join(base_dir, ".env"))

from pipeline.blip_model import run_blip
from pipeline.bert_model import run_bert
from pipeline.search import search_news
from pipeline.utils import text_cleaning

# Fake News Sample
# From: http://rightwingnews.com/top-news/famous-dog-killed-spot-waited-year-owner-return/
headline = "Famous dog killed in spot she waited a year for her owner to return to!"
caption = "Loyal Dog Loung awaited owners by Thai roadside for a YEAR."

article_text = """You know what? Don’t even read further if you are part of the man’s best friend club, 
you just might regret your decision to get out of bed this morning. News just hit the web that the locally 
famous three year-old dog named Loung, who fell out of a truck last year, landing on a busy route between 
Cambodia and Thailand, has been hit and killed by a car as it waited for the return of its owner for over a year.
You can start the waterworks as we go into the details. The Sukhumvit Road, or Thailand Route 3, is a leading highway 
between Bangkok and Cambodia and Loung’s owner must not have realized that she had fallen right out of the truck. 
What’s truly amazing is that the dog refused to leave the area, hoping that one day her owner would return and rescue her."""

image_path = os.path.join(script_dir, "fakenews.jpg")
result_dir = os.path.join(script_dir, "llm_eval_results")

# Prompt Engineered Input
def build_prompt(headline, caption, article_text, blip_results, bert_results, search_results):
    sources_text = ""
    for i, result in enumerate(search_results, 1):
        sources_text += f"\nSource {i}: {result['title']}\nURL: {result['url']}\nSnippet: {result['snippet']}\n"

    prompt = f"""
        You are an educational fake news detection assistant helping general news consumers 
        understand and critically evaluate news content.

        You have been provided with the following article and information:

        ### ARTICLE HEADLINE ###
        {headline}

        ### IMAGE CAPTION ###
        {caption}

        ## ARTICLE TEXT ###
        {article_text}

        ### VISUAL ANALYSIS (BLIP Image-Text Matching) ###
        Image-Text Match Probability: {blip_results['match_prob']}
        Image-Text Mismatch Probability: {blip_results['mismatch_prob']}

        ### TEXTUAL ANALYSIS (RoBERTa Fake News Detection) ###
        Fake Probability: {bert_results['fake_prob']}
        Real Probability: {bert_results['real_prob']}
        Prediction: {'FAKE' if bert_results['prediction'] == 0 else 'REAL'}

        ### RETRIEVED SOURCES ###
        {sources_text}

        ### YOUR TASK ###
        Generate a clear, neutral and educational report structured as follows:

        1. OVERALL VERDICT
        Provide an overall assessment of the content based on the analysis results.

        2. TEXTUAL ANALYSIS FINDINGS
        Explain what the textual analysis found and what it means for the credibility 
        of the article. Use the fake and real probabilities to support your explanation.

        3. VISUAL ANALYSIS FINDINGS
        Explain what the visual analysis found regarding the relationship between the 
        image and the article text. Use the match and mismatch probabilities to support 
        your explanation.

        4. SUPPORTING EVIDENCE
        Using ONLY the retrieved sources provided above, provide context and evidence 
        that supports or contradicts the claims made in the article. 
        Cite each source by name and URL. Provide all URLs in order.

        5. EDUCATIONAL TAKEAWAY
        Provide a brief educational summary helping the user understand how to identify 
        similar misinformation in the future.

        ### CONSTRAINTS ###
        - Be neutral and educational, never accusatory or sensationalist
        - Only reference sources from the retrieved sources provided above
        - Do not fabricate or hallucinate any sources or facts
        - Use clear, simple language suitable for a general audience
        - Always cite sources by name and URL when referencing them
        """
    return prompt

# Calling the models
# GEMINI
def call_gemini(prompt):
    from google import genai
    client = genai.Client(api_key = os.getenv("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model = "gemini-2.5-flash",
        contents = prompt
    )
    return response.text

# MISTRAL
def call_mistral(prompt):
    from mistralai import Mistral
    client = Mistral(api_key = os.getenv("MISTRAL_API_KEY"))
    response = client.chat.complete(
        model = "mistral-large-latest",
        messages = [{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# GROQ (LLAMA)
def call_groq(prompt):
    from groq import Groq
    client = Groq(api_key = os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model = "llama-3.3-70b-versatile",
        messages = [{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Saving Results
def save_results(name, prompt, response):

    os.makedirs(result_dir, exist_ok = True)
    
    with open(os.path.join(result_dir, f"{name}.txt"), "w", encoding = "utf-8") as f:
        f.write(f"======PROMPT======\n{prompt}\n\n")
        f.write(f"======RESPONSE======\n{response}\n")
    print(f"Saved {name} response.")

# Main Script Pipeline
def main():
    print("=" * 30)
    print("LLM Evaluation Script")

    # Load test image
    print("\nLoading test image")
    image = Image.open(image_path).convert("RGB")
    print("Image loaded.")

    # Run BLIP Model
    print("\nRunning BLIP")
    blip_results = run_blip(image, caption)
    print(f"BLIP Results: {blip_results}")

    # Run BERT Model
    print("\nRunning BERT")
    cleaned_text = text_cleaning(article_text)
    bert_results = run_bert(cleaned_text)
    print(f"BERT Results: {bert_results}")

    # Run DuckDuckGo Search
    print("\nRunning DuckDuckGo Search")
    search_results = search_news(headline, bert_results, blip_results)
    print(f"Search Results:\n{json.dumps(search_results, indent=2)}")

    # Build prompt for LLMs
    print("\nBuilding prompt")
    prompt = build_prompt(headline, caption, article_text, blip_results, bert_results, search_results)

    # Call all three LLMs
    # GEMINI
    print("\nCalling Gemini 2.5 Flash...")
    gemini_response = call_gemini(prompt)
    save_results("gemini_2_5_flash", prompt, gemini_response)

    # MISTRAL
    print("Calling Mistral...")
    mistral_response = call_mistral(prompt)
    save_results("mistral_large", prompt, mistral_response)

    # GROQ (LLAMA)
    print("Calling Groq Llama 3.3 70B...")
    groq_response = call_groq(prompt)
    save_results("groq_llama3_70b", prompt, groq_response)

    print("All LLM evaluations complete.")
    print(f"Results saved to {result_dir}")
    print("=" * 30)

main()