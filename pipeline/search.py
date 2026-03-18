"""
search.py File
Search module of the application
Crafts targetted search queries from model outputs
and returns relevant news articles using DuckDuckGo
"""

# Dependencies
from ddgs import DDGS

# Dynamic Query Crafting
# Input: Headline, Bert_Results, Blip_Results
# Output: Crafted Query
def craft_query(headline, bert_results, blip_results):
    query = headline

    # If Bert predicted fake
    if bert_results["prediction"] == 0:
        query = f"fact check {query}"
    
    # If blip mismatch probability is higher than 0.5
    if blip_results["mismatch_prob"] >= 0.5:
        query = f"{query} image misuse"

    return query

# Search Function
def search_news(headline, bert_results, blip_results):
    query = craft_query(headline, bert_results, blip_results)
    results = []

    # Running the search
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results = 5):
            results.append({
                "title": r["title"],
                "url": r["href"],
                "snippet": r["body"]
            })

    return results