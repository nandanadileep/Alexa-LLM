"""
Smoke tests — happy path only. Run these for a quick sanity check.
For full coverage run test_lambda.py instead.
"""

import os
import unittest
from unittest.mock import patch, MagicMock

os.environ.setdefault("LLM_PROVIDER", "groq")
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("DYNAMODB_TABLE", "TestTable")

import lambda_function


def ask_event(query="what is the meaning of life"):
    return {
        "version": "1.0",
        "session": {"user": {"userId": "u1"}, "attributes": {}},
        "request": {
            "type": "IntentRequest",
            "intent": {
                "name": "AskClaudeIntent",
                "slots": {"query": {"value": query}},
            },
        },
    }


class SmokeTests(unittest.TestCase):

    def test_launch(self):
        event = {"version": "1.0", "session": {"user": {"userId": "u1"}}, "request": {"type": "LaunchRequest"}}
        r = lambda_function.lambda_handler(event, None)
        self.assertEqual(r["version"], "1.0")
        self.assertFalse(r["response"]["shouldEndSession"])

    @patch("lambda_function.get_user_context", return_value="")
    @patch("lambda_function.save_history")
    @patch("lambda_function.get_history", return_value=[])
    @patch("lambda_function.ask_llm", return_value="42.")
    def test_ask(self, mock_llm, *_):
        r = lambda_function.lambda_handler(ask_event(), None)
        self.assertEqual(r["response"]["outputSpeech"]["text"], "42.")

    def test_stop(self):
        event = {"version": "1.0", "session": {"user": {"userId": "u1"}}, "request": {"type": "IntentRequest", "intent": {"name": "AMAZON.StopIntent", "slots": {}}}}
        r = lambda_function.lambda_handler(event, None)
        self.assertTrue(r["response"]["shouldEndSession"])

    @patch("lambda_function.save_user_context")
    def test_set_context(self, mock_save):
        event = {"version": "1.0", "session": {"user": {"userId": "u1"}}, "request": {"type": "IntentRequest", "intent": {"name": "SetContextIntent", "slots": {"context": {"value": "I am a chef"}}}}}
        r = lambda_function.lambda_handler(event, None)
        mock_save.assert_called_once_with("u1", "I am a chef")
        self.assertIn("Got it", r["response"]["outputSpeech"]["text"])

    @patch("lambda_function.get_user_context", return_value="")
    @patch("lambda_function.save_history")
    @patch("lambda_function.get_history", return_value=[])
    @patch("lambda_function.ask_llm", side_effect=Exception("boom"))
    def test_llm_failure_graceful(self, *_):
        r = lambda_function.lambda_handler(ask_event(), None)
        self.assertIn("trouble", r["response"]["outputSpeech"]["text"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
