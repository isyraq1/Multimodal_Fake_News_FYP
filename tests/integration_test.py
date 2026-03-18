"""
integration_tests.py File
This file contains the various integration testing carried out
"""

# Dependencies
import os
import sys
import unittest
from PIL import Image
from dotenv import load_dotenv

# Env & Path Setup
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
load_dotenv(os.path.join(base_dir, ".env"))

from pipeline.blip_model import run_blip
from pipeline.bert_model import run_bert
from pipeline.search import craft_query, search_news
from pipeline.llm import build_prompt, generate_report
from pipeline.utils import text_cleaning

# Test Data: LLM Evaluation Data Truncated
test_headline = "Famous dog killed in spot she waited a year for her owner to return to"
test_caption = "Loyal dog waits by roadside for owner to return"
test_article = """A locally famous dog named Loung who fell out of a truck last year 
                has been hit and killed by a car as it waited for the return of its 
                owner for over a year."""

# Mock Image
test_image = Image.new("RGB", (224, 224), color = (255, 255, 255))

### Integration Test 1: BLIP + BERT => Search Module ###
class TestModelsToSearch(unittest.TestCase):

    # Setup run to retrieve outputs
    def setUp(self):
        cleaned_text = text_cleaning(test_article)
        self.blip_results = run_blip(test_image, test_caption)
        self.bert_results = run_bert(cleaned_text)

    # Verifies that the BLIP output feeds into the query
    def test_blip_output_feeds_into_query(self):
        query = craft_query(test_headline, self.bert_results, self.blip_results)
        self.assertIsInstance(query, str)
        self.assertGreater(len(query), 0)

    # Verifies that the BERT prediction adjusts the query
    def test_bert_output_feeds_into_query(self):
        query = craft_query(test_headline, self.bert_results, self.blip_results)
        if self.bert_results["prediction"] == 0:
            self.assertIn("fact check", query.lower())

    # Verifies that both model outputs feed into the search correctly
    def test_model_outputs_feed_into_search(self):
        results = search_news(test_headline, self.bert_results, self.blip_results)
        self.assertIsInstance(results, list)

### Integration Test 2: BLIP + BERT => Search => LLM
class TestModelsAndSearchToLLM(unittest.TestCase):

    # Setup run to retrieve outputs
    def setUp(self):
        cleaned_text = text_cleaning(test_article)
        self.blip_results = run_blip(test_image, test_caption)
        self.bert_results = run_bert(cleaned_text)
        self.search_results = search_news(test_headline, self.bert_results, self.blip_results)

    # Verifies that the output feeds into build_prompt function
    def test_all_outputs_feed_into_build_prompt(self):
        prompt = build_prompt(
            test_headline, 
            test_caption, 
            test_article,
            self.blip_results, 
            self.bert_results, 
            self.search_results
        )

        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 0)

    # Verifies that the built prompt has model scores
    def test_prompt_contains_model_scores(self):
        prompt = build_prompt(
            test_headline, 
            test_caption, 
            test_article,
            self.blip_results, 
            self.bert_results, 
            self.search_results
        )

        self.assertIn(str(self.blip_results["match_prob"]), prompt)
        self.assertIn(str(self.bert_results["fake_prob"]), prompt)

    # Verifies that an output (string) is provided from generate_report function
    def test_generate_report_returns_output(self):
        report = generate_report(
            test_headline, 
            test_caption, 
            test_article,
            self.blip_results, 
            self.bert_results, 
            self.search_results
        )

        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 0)

### Integration Test 3: Full Pipeline from input to report
class TestFullPipeline(unittest.TestCase):

    # Tests the raw inputs on the full pipeline
    def test_full_pipeline_process(self):
        cleaned_text = text_cleaning(test_article)
        blip_results = run_blip(test_image, test_caption)
        bert_results = run_bert(cleaned_text)
        search_results = search_news(test_headline, bert_results, blip_results)
        report = generate_report(
            test_headline, 
            test_caption, 
            test_article,
            blip_results, 
            bert_results, 
            search_results
        )

        # Verifies no empty outputs for each module
        self.assertIsInstance(blip_results, dict)
        self.assertIsInstance(bert_results, dict)
        self.assertIsInstance(search_results, list)
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 0)

    # Verifies that the end report contains the correct sections
    def test_full_pipeline_report_contains_sections(self):
        # Verify report contains expected sections
        cleaned_text    = text_cleaning(test_article)
        blip_results    = run_blip(test_image, test_caption)
        bert_results    = run_bert(cleaned_text)
        search_results  = search_news(test_headline, bert_results, blip_results)
        report          = generate_report(
            test_headline, test_caption, test_article,
            blip_results, bert_results, search_results
        )

        # Verifies that the 5 sections are present
        self.assertIn("overall verdict", report.lower())
        self.assertIn("textual analysis", report.lower())
        self.assertIn("visual analysis", report.lower())
        self.assertIn("supporting evidence", report.lower())
        self.assertIn("educational takeaway", report.lower())

### Run the Tests ###
if __name__ == "__main__":
    unittest.main(verbosity = 2)