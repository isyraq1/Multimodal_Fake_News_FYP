"""
llm.py File
Module for running the results through an LLM
Generates a proper report for users to understand
And read, providing valid advise for the inputs provided.
"""

# Dependencies
import os
from mistralai import Mistral

# Variable Configuration
mistral_model = "mistral-large-latest"

# Prompt Engineering Function
# Inputs: headline, caption, article_text, blip_results, bert_results, search_results
# Output: LLM Input Prompt
def build_prompt(headline, caption, article_text, blip_results, bert_results, search_results):

    # Format Search Results
    sources_text = ""
    for i, result in enumerate(search_results, 1):
        sources_text += f"\nSource {i}: {result['title']}\nURL: {result['url']}\nSnippet: {result['snippet']}\n"

    prompt = f"""
        You are an educational fake news detection assitant helping general news consumers
        understand and critically evaluate news content.

        You have been provided with the following input information:

        ### ARTICLE HEADLINE ###
        {headline}

        ### ARTICLE TEXT ###
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
        Provide an overall assessment based on all analysis results.
        Note that a high image-text match score only confirms semantic/topical
        consistency, not the truthfulness of the claims

        2. TEXTUAL ANALYSIS FINDINGS
        Explain what the textual analysis found and what it means for the
        credibility of the article. Use the fake and real probabilities to support
        your explanation.

        3. VISUAL ANALYSIS FINDINGS
        Explain what the visual analysis found regarding the relationship
        between the image and the article text. Clarify that a high match
        score confirms topical and semantic consistency only, but not truthfulness.
        If image mismatch is detected, explain specifically that images can be taken
        from unrelated events and paired with false narratives to attempt
        to misled readers.
        
        4. SUPPORTING EVIDENCE
        Using ONLY the retrieved sources provided above, analyse them and provide
        context and evidence that directly supports or contradicts the claims
        made in the article. Note the credibility and type of each source, for example
        whether it is a fact-checking site, mainstream news outlet or unverified blog.
        Critically assess whether the sources corroborate or merely recycle old
        content. Check source dates where possible. If no relevant sources are found,
        simply state that no relevant sources were retrieved and do not list any sources.
        Cite each source by name and URL.
        
        5. EDUCATIONAL TAKEAWAY
        Provide a comprehensive educational report helping the user identify similar
        misinformation in the future. Cover topics like source verification, publication
        dates, emotional manipulation and fact-checking tools. If the article is deemed to be
        fully true, inclusive of visual and textual information, suggest good practices to uphold.

        ### CONSTRAINTS ###
        - Be neutral and educational, never accusatory or sensationalist.
        - Only reference sources from the retrieved sources provided above.
        - Do not fabricate or hallucinate any sources or facts.
        - Use clear simple language suitable for a general audience.
        - Always cite sources by name and URL when referencing them
        - If retrieved sources appear to support claims, critically assess 
        whether they are genuine corroboration or recycled content.
        - Prioritise model analysis outputs over source content when forming 
        the overall verdict.
        - Keep the report under 650 words total.
        - Before generating the report, reason through the evidence 
        systematically by considering the model scores, source dates, source 
        credibility and content consistency together.
        - Express findings with appropriate confidence. Use language like 
        'suggests', 'indicates' or 'appears' rather than absolute statements.
        - If the article is deemed fully credible, provide positive 
        reinforcement and suggest good practices for media consumption.
    """

    return prompt

# Generate Report Function
# Inputs: headline, caption, article_text, blip_results, bert_results, search_results
# Output: LLM Input Prompt
def generate_report(headline, caption, article_text, blip_results, bert_results, search_results):
    
    # Connecting to client
    client = Mistral(api_key = os.getenv("MISTRAL_API_KEY"))
    
    # Generating prompt
    prompt = build_prompt(headline, caption, article_text, blip_results, bert_results, search_results)
    
    # Passing prompt and retrieving response
    response = client.chat.complete(
        model = mistral_model,
        messages = [{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content