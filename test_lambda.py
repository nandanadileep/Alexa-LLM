import json
import os
import unittest
from unittest.mock import patch, MagicMock

os.environ["LLM_PROVIDER"] = "openrouter"
os.environ["OPENROUTER_API_KEY"] = "test-key"
os.environ["DYNAMODB_TABLE"] = "AlexaConversationHistory"

import lambda_function


def make_event(request_type, intent_name=None, query=None, session_new=True):
    event = {
        "version": "1.0",
        "session": {
            "new": session_new,
            "sessionId": "test-session-id",
            "application": {"applicationId": "test-app-id"},
            "attributes": {},
            "user": {"userId": "test-user-id"},
        },
        "context": {
            "System": {
                "application": {"applicationId": "test-app-id"},
                "user": {"userId": "test-user-id"},
                "device": {"deviceId": "test-device-id", "supportedInterfaces": {}},
                "apiEndpoint": "https://api.amazonalexa.com",
                "apiAccessToken": "test-token",
            }
        },
        "request": {"type": request_type},
    }

    if request_type == "IntentRequest":
        event["request"]["intent"] = {
            "name": intent_name,
            "confirmationStatus": "NONE",
            "slots": {},
        }
        if query:
            event["request"]["intent"]["slots"]["query"] = {
                "name": "query",
                "value": query,
                "confirmationStatus": "NONE",
            }

    if request_type == "SessionEndedRequest":
        event["request"]["reason"] = "USER_INITIATED"

    return event


class TestLambdaHandler(unittest.TestCase):

    def test_launch_request(self):
        event = make_event("LaunchRequest")
        response = lambda_function.lambda_handler(event, None)
        self.assertEqual(response["version"], "1.0")
        self.assertIn("AI-powered assistant", response["response"]["outputSpeech"]["text"])
        self.assertFalse(response["response"]["shouldEndSession"])

    def test_help_intent(self):
        event = make_event("IntentRequest", intent_name="AMAZON.HelpIntent")
        response = lambda_function.lambda_handler(event, None)
        self.assertIn("question", response["response"]["outputSpeech"]["text"])

    def test_stop_intent(self):
        event = make_event("IntentRequest", intent_name="AMAZON.StopIntent")
        response = lambda_function.lambda_handler(event, None)
        self.assertEqual(response["response"]["outputSpeech"]["text"], "Goodbye!")
        self.assertTrue(response["response"]["shouldEndSession"])

    def test_cancel_intent(self):
        event = make_event("IntentRequest", intent_name="AMAZON.CancelIntent")
        response = lambda_function.lambda_handler(event, None)
        self.assertTrue(response["response"]["shouldEndSession"])

    def test_fallback_intent(self):
        event = make_event("IntentRequest", intent_name="AMAZON.FallbackIntent")
        response = lambda_function.lambda_handler(event, None)
        self.assertIn("didn't understand", response["response"]["outputSpeech"]["text"])

    def test_unknown_intent(self):
        event = make_event("IntentRequest", intent_name="SomeUnknownIntent")
        response = lambda_function.lambda_handler(event, None)
        self.assertIn("didn't understand", response["response"]["outputSpeech"]["text"])

    def test_session_ended_request(self):
        event = make_event("SessionEndedRequest")
        response = lambda_function.lambda_handler(event, None)
        self.assertTrue(response["response"]["shouldEndSession"])

    @patch("lambda_function.save_history")
    @patch("lambda_function.get_history", return_value=[])
    @patch("lambda_function.ask_llm", return_value="Marcus Aurelius wrote Meditations as a personal journal.")
    def test_ask_intent_success(self, mock_llm, mock_get, mock_save):
        event = make_event("IntentRequest", intent_name="AskClaudeIntent", query="who is marcus aurelius")
        response = lambda_function.lambda_handler(event, None)
        self.assertEqual(response["response"]["outputSpeech"]["text"], "Marcus Aurelius wrote Meditations as a personal journal.")
        mock_llm.assert_called_once_with("who is marcus aurelius", [])
        mock_save.assert_called_once()

    @patch("lambda_function.save_history")
    @patch("lambda_function.get_history", return_value=[])
    @patch("lambda_function.ask_llm", side_effect=Exception("API error"))
    def test_ask_intent_llm_failure(self, mock_llm, mock_get, mock_save):
        event = make_event("IntentRequest", intent_name="AskClaudeIntent", query="test question")
        response = lambda_function.lambda_handler(event, None)
        self.assertIn("trouble", response["response"]["outputSpeech"]["text"])
        mock_save.assert_not_called()

    @patch("lambda_function.save_history")
    @patch("lambda_function.get_history", return_value=[])
    def test_ask_intent_no_query(self, mock_get, mock_save):
        event = make_event("IntentRequest", intent_name="AskClaudeIntent")
        response = lambda_function.lambda_handler(event, None)
        self.assertIn("didn't catch", response["response"]["outputSpeech"]["text"])
        mock_save.assert_not_called()

    @patch("lambda_function.save_history")
    @patch("lambda_function.get_history", return_value=[
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ])
    @patch("lambda_function.ask_llm", return_value="Yes, Stoicism is a great philosophy.")
    def test_ask_intent_with_history(self, mock_llm, mock_get, mock_save):
        event = make_event("IntentRequest", intent_name="AskClaudeIntent", query="tell me more")
        lambda_function.lambda_handler(event, None)
        history_passed = mock_llm.call_args[0][1]
        self.assertEqual(len(history_passed), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
