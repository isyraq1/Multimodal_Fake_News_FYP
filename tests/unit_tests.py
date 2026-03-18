"""
unit_tests.py File
This file contains all the unittest conducted for the application
"""

# Dependencies
import os
import sys
import unittest
from PIL import Image

# Path Configuration
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from pipeline.utils import text_cleaning
from pipeline.blip_model import run_blip
from pipeline.bert_model import run_bert
from pipeline.search import craft_query, search_news
from pipeline.llm import build_prompt

# Mock Data for Testing (Adapted from LLM Eval)
mock_headline = "Famous dog killed in spot she waited a year for her owner to return to"
mock_caption = "Loyal dog waits by roadside for owner to return"
mock_article = "A locally famous dog named Loung who fell out of a truck last year has been hit and killed by a car."
mock_image = Image.new("RGB", (224, 224), color = (255, 255, 255))

mock_blip_fake = {"match_prob": 0.15, "mismatch_prob": 0.85}
mock_blip_real = {"match_prob": 0.90, "mismatch_prob": 0.10}
mock_bert_fake = {"fake_prob": 0.85, "real_prob": 0.15, "prediction": 0}
mock_bert_real = {"fake_prob": 0.05, "real_prob": 0.95, "prediction": 1}

mock_source = [
    {
        "title": "Dog story fake.",
        "url": "https://example.com",
        "snippet": "The story has been found to be fake."
    }
]

### Text Cleaning Test ###
class TestTextCleaning(unittest.TestCase):

    # URL Removal
    def test_removes_urls(self):
        text = "Check this out https://example.com for more info"
        result = text_cleaning(text)
        self.assertNotIn("https://", result)

    # Extra white space removal
    def test_removes_extra_whitespace(self):
        text = "Extra  White  Space"
        result = text_cleaning(text)
        self.assertEqual(result, "Extra White Space")

    # Empty string test
    def test_empty_string(self):
        result = text_cleaning("")
        self.assertEqual(result, "")

### BLIP Model Tests ###
class TestRunBlip(unittest.TestCase):

    # Return dictionary
    def test_returns_dict(self):
        result = run_blip(mock_image, mock_caption)
        self.assertIsInstance(result, dict)

    # Returns appropriate keys in dict
    def test_contains_required_keys(self):
        result = run_blip(mock_image, mock_caption)
        self.assertIn("match_prob", result)
        self.assertIn("mismatch_prob", result)

    # Returned probabilities are in float
    def test_probabilities_are_floats(self):
        result = run_blip(mock_image, mock_caption)
        self.assertIsInstance(result["match_prob"], float)
        self.assertIsInstance(result["mismatch_prob"], float)

    # Returned valid scores between 0 and 1
    def test_probabilities_between_zero_and_one(self):
        result = run_blip(mock_image, mock_caption)
        self.assertGreaterEqual(result["match_prob"], 0.0)
        self.assertLessEqual(result["match_prob"], 1.0)
        self.assertGreaterEqual(result["mismatch_prob"], 0.0)
        self.assertLessEqual(result["mismatch_prob"], 1.0)

    # Returned probilities sum total to one
    def test_probabilities_sum_to_one(self):
        result = run_blip(mock_image, mock_caption)
        total = round(result["match_prob"] + result["mismatch_prob"], 2)
        self.assertAlmostEqual(total, 1.0, places=1)

### BERT Model Tests ###
class TestRunBert(unittest.TestCase):

    # Outputs a dictionary
    def test_returns_dict(self):
        result = run_bert(mock_article)
        self.assertIsInstance(result, dict)

    # Returns the correct keys in the dict
    def test_contains_required_keys(self):
        result = run_bert(mock_article)
        self.assertIn("fake_prob", result)
        self.assertIn("real_prob", result)
        self.assertIn("prediction", result)

    # Returns the keys as float
    def test_probabilities_are_floats(self):
        result = run_bert(mock_article)
        self.assertIsInstance(result["fake_prob"], float)
        self.assertIsInstance(result["real_prob"], float)

    # Checks that all returned values are between 0 and 1
    def test_probabilities_between_zero_and_one(self):
        result = run_bert(mock_article)
        self.assertGreaterEqual(result["fake_prob"], 0.0)
        self.assertLessEqual(result["fake_prob"], 1.0)
        self.assertGreaterEqual(result["real_prob"], 0.0)
        self.assertLessEqual(result["real_prob"], 1.0)

    # Checks that the final prediction is a binary 0 or 1
    def test_prediction_is_binary(self):
        result = run_bert(mock_article)
        self.assertIn(result["prediction"], [0, 1])

### Search Module Query Test ###
class TestCraftQuery(unittest.TestCase):

    # Checks if fact check is added to query
    def test_adds_fact_check_when_bert_fake(self):
        query = craft_query(mock_headline, mock_bert_fake, mock_blip_real)
        self.assertIn("fact check", query.lower())

    # Checks if image misuse if added to query
    def test_adds_image_misuse_when_blip_mismatch(self):
        query = craft_query(mock_headline, mock_bert_real, mock_blip_fake)
        self.assertIn("image misuse", query.lower())

    # Checks image misuse and fact check added to query
    def test_adds_both_when_both_flagged(self):
        query = craft_query(mock_headline, mock_bert_fake, mock_blip_fake)
        self.assertIn("fact check", query.lower())
        self.assertIn("image misuse", query.lower())

    # Checks if headline is set as the query when both real
    def test_returns_headline_only_when_both_real(self):
        query = craft_query(mock_headline, mock_bert_real, mock_blip_real)
        self.assertEqual(query, mock_headline)

### Search Module Search Test ###
class TestSearchNews(unittest.TestCase):

    # Checks if the returned output is a list
    def test_returns_list(self):
        results = search_news(mock_headline, mock_bert_fake, mock_blip_real)
        self.assertIsInstance(results, list)

    # Checks if the returned lists has the correct keys
    def test_results_contain_required_keys(self):
        results = search_news(mock_headline, mock_bert_fake, mock_blip_real)
        if results:
            self.assertIn("title", results[0])
            self.assertIn("url", results[0])
            self.assertIn("snippet", results[0])

### LLM Model Test ###
class TestBuildPrompt(unittest.TestCase):

    # Checks if test returns a string
    def test_returns_string(self):
        result = build_prompt(
            mock_headline, 
            mock_caption, 
            mock_article,
            mock_blip_fake, 
            mock_bert_fake, 
            mock_source
        )
        self.assertIsInstance(result, str)

    # Tests that prompt build is not empty
    def test_prompt_not_empty(self):
        result = build_prompt(
            mock_headline, 
            mock_caption, 
            mock_article,
            mock_blip_fake, 
            mock_bert_fake, 
            mock_source
        )
        self.assertGreater(len(result), 0)

    # Tests if promopt includes the article headline
    def test_prompt_contains_headline(self):
        result = build_prompt(
            mock_headline, 
            mock_caption, 
            mock_article,
            mock_blip_fake, 
            mock_bert_fake, 
            mock_source
        )
        self.assertIn(mock_headline, result)

### Run Tests ###
if __name__ == "__main__":
    unittest.main(verbosity = 2)