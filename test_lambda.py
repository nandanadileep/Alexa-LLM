"""
Comprehensive test suite for Alexa-LLM Lambda function.

Covers:
  - lambda_function: all intent handlers, routing, response builder
  - groq_provider:   API success, errors, prompt construction
  - gemini_provider: API success, errors, role mapping, prompt construction
  - openrouter_provider: API success, errors, prompt construction
  - dynamo:          all CRUD paths, missing items, env-var table name
"""

import io
import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch, call

# ── Environment must be set before any project import ──────────────────────────
os.environ.setdefault("LLM_PROVIDER", "groq")
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")
os.environ.setdefault("OPENROUTER_API_KEY", "test-openrouter-key")
os.environ.setdefault("DYNAMODB_TABLE", "TestTable")

import lambda_function


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def make_event(
    request_type,
    intent_name=None,
    query=None,
    question=None,
    context_value=None,
    session_new=True,
    user_id="test-user-id",
    session_reason=None,
    session_attributes=None,
):
    """Build a minimal but valid Alexa request event."""
    event = {
        "version": "1.0",
        "session": {
            "new": session_new,
            "sessionId": "amzn1.echo-api.session.test",
            "application": {"applicationId": "amzn1.ask.skill.test"},
            "attributes": session_attributes or {},
            "user": {"userId": user_id},
        },
        "context": {
            "System": {
                "application": {"applicationId": "amzn1.ask.skill.test"},
                "user": {"userId": user_id},
                "device": {
                    "deviceId": "amzn1.ask.device.test",
                    "supportedInterfaces": {},
                },
                "apiEndpoint": "https://api.amazonalexa.com",
                "apiAccessToken": "test-token",
            }
        },
        "request": {"type": request_type, "requestId": "amzn1.echo-api.request.test"},
    }

    if request_type == "IntentRequest":
        slots = {}
        if query is not None:
            slots["query"] = {"name": "query", "value": query, "confirmationStatus": "NONE"}
        if question is not None:
            slots["question"] = {"name": "question", "value": question, "confirmationStatus": "NONE"}
        if context_value is not None:
            slots["context"] = {"name": "context", "value": context_value, "confirmationStatus": "NONE"}
        event["request"]["intent"] = {
            "name": intent_name,
            "confirmationStatus": "NONE",
            "slots": slots,
        }

    if request_type == "SessionEndedRequest":
        event["request"]["reason"] = session_reason or "USER_INITIATED"

    return event


def get_speech(response):
    return response["response"]["outputSpeech"]["text"]


def get_reprompt(response):
    return response["response"].get("reprompt", {}).get("outputSpeech", {}).get("text")


def make_urlopen_mock(body: dict, status: int = 200):
    """Return a context-manager mock that yields a fake HTTP response."""
    encoded = json.dumps(body).encode("utf-8")
    mock_response = MagicMock()
    mock_response.read.return_value = encoded
    mock_response.status = status
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = MagicMock(return_value=False)
    return mock_response


# ══════════════════════════════════════════════════════════════════════════════
# 1. build_response
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildResponse(unittest.TestCase):

    def test_basic_structure(self):
        r = lambda_function.build_response("Hello")
        self.assertEqual(r["version"], "1.0")
        self.assertIn("response", r)
        self.assertIn("sessionAttributes", r)

    def test_speech_text(self):
        r = lambda_function.build_response("Hello there")
        self.assertEqual(get_speech(r), "Hello there")

    def test_plain_text_type(self):
        r = lambda_function.build_response("Hi")
        self.assertEqual(r["response"]["outputSpeech"]["type"], "PlainText")

    def test_session_stays_open_by_default(self):
        r = lambda_function.build_response("Hi")
        self.assertFalse(r["response"]["shouldEndSession"])

    def test_session_ends_when_requested(self):
        r = lambda_function.build_response("Bye", should_end=True)
        self.assertTrue(r["response"]["shouldEndSession"])

    def test_reprompt_present_when_session_open(self):
        r = lambda_function.build_response("Hi")
        self.assertIsNotNone(get_reprompt(r))
        self.assertTrue(len(get_reprompt(r)) > 0)

    def test_no_reprompt_when_session_ends(self):
        r = lambda_function.build_response("Bye", should_end=True)
        self.assertIsNone(get_reprompt(r))

    def test_custom_session_attributes(self):
        attrs = {"foo": "bar", "count": 3}
        r = lambda_function.build_response("Hi", session_attributes=attrs)
        self.assertEqual(r["sessionAttributes"], attrs)

    def test_default_session_attributes_are_empty_dict(self):
        r = lambda_function.build_response("Hi")
        self.assertEqual(r["sessionAttributes"], {})

    def test_empty_speech_string(self):
        r = lambda_function.build_response("")
        self.assertEqual(get_speech(r), "")


# ══════════════════════════════════════════════════════════════════════════════
# 2. LaunchRequest
# ══════════════════════════════════════════════════════════════════════════════

class TestLaunchRequest(unittest.TestCase):

    def test_returns_greeting(self):
        event = make_event("LaunchRequest")
        r = lambda_function.lambda_handler(event, None)
        self.assertEqual(r["version"], "1.0")
        self.assertFalse(r["response"]["shouldEndSession"])

    def test_greeting_is_non_empty(self):
        event = make_event("LaunchRequest")
        r = lambda_function.lambda_handler(event, None)
        self.assertTrue(len(get_speech(r)) > 0)

    def test_session_stays_open(self):
        event = make_event("LaunchRequest")
        r = lambda_function.lambda_handler(event, None)
        self.assertFalse(r["response"]["shouldEndSession"])


# ══════════════════════════════════════════════════════════════════════════════
# 3. AskClaudeIntent
# ══════════════════════════════════════════════════════════════════════════════

class TestAskIntent(unittest.TestCase):

    @patch("lambda_function.get_user_context", return_value="")
    @patch("lambda_function.save_history")
    @patch("lambda_function.get_history", return_value=[])
    @patch("lambda_function.ask_llm", return_value="Stoicism is a philosophy of resilience.")
    def test_success_with_query_slot(self, mock_llm, mock_get, mock_save, mock_ctx):
        event = make_event("IntentRequest", intent_name="AskClaudeIntent", query="what is stoicism")
        r = lambda_function.lambda_handler(event, None)
        self.assertEqual(get_speech(r), "Stoicism is a philosophy of resilience.")
        mock_llm.assert_called_once_with("what is stoicism", [], "")

    @patch("lambda_function.get_user_context", return_value="")
    @patch("lambda_function.save_history")
    @patch("lambda_function.get_history", return_value=[])
    @patch("lambda_function.ask_llm", return_value="Stoicism is about self-control.")
    def test_success_with_question_slot(self, mock_llm, mock_get, mock_save, mock_ctx):
        event = make_event("IntentRequest", intent_name="AskClaudeIntent", question="tell me about stoicism")
        r = lambda_function.lambda_handler(event, None)
        self.assertEqual(get_speech(r), "Stoicism is about self-control.")
        mock_llm.assert_called_once_with("tell me about stoicism", [], "")

    @patch("lambda_function.get_user_context", return_value="")
    @patch("lambda_function.save_history")
    @patch("lambda_function.get_history", return_value=[])
    def test_no_query_returns_prompt(self, mock_get, mock_save, mock_ctx):
        event = make_event("IntentRequest", intent_name="AskClaudeIntent")
        r = lambda_function.lambda_handler(event, None)
        self.assertIn("didn't catch", get_speech(r))
        mock_save.assert_not_called()

    @patch("lambda_function.get_user_context", return_value="")
    @patch("lambda_function.save_history")
    @patch("lambda_function.get_history", return_value=[])
    @patch("lambda_function.ask_llm", side_effect=Exception("API timeout"))
    def test_llm_failure_returns_error_message(self, mock_llm, mock_get, mock_save, mock_ctx):
        event = make_event("IntentRequest", intent_name="AskClaudeIntent", query="hello")
        r = lambda_function.lambda_handler(event, None)
        self.assertIn("trouble", get_speech(r))

    @patch("lambda_function.get_user_context", return_value="")
    @patch("lambda_function.save_history")
    @patch("lambda_function.get_history", return_value=[])
    @patch("lambda_function.ask_llm", side_effect=Exception("API timeout"))
    def test_llm_failure_does_not_save_history(self, mock_llm, mock_get, mock_save, mock_ctx):
        event = make_event("IntentRequest", intent_name="AskClaudeIntent", query="hello")
        lambda_function.lambda_handler(event, None)
        mock_save.assert_not_called()

    @patch("lambda_function.get_user_context", return_value="")
    @patch("lambda_function.save_history")
    @patch("lambda_function.get_history", return_value=[
        {"role": "user", "content": "what is stoicism"},
        {"role": "assistant", "content": "Stoicism is a philosophy."},
    ])
    @patch("lambda_function.ask_llm", return_value="Epictetus was a Stoic philosopher.")
    def test_history_passed_to_llm(self, mock_llm, mock_get, mock_save, mock_ctx):
        event = make_event("IntentRequest", intent_name="AskClaudeIntent", query="who is epictetus")
        lambda_function.lambda_handler(event, None)
        _, history_arg, _ = mock_llm.call_args[0]
        self.assertEqual(len(history_arg), 2)
        self.assertEqual(history_arg[0]["role"], "user")

    @patch("lambda_function.get_user_context", return_value="")
    @patch("lambda_function.save_history")
    @patch("lambda_function.get_history", return_value=[])
    @patch("lambda_function.ask_llm", return_value="Sure.")
    def test_history_updated_with_new_turn(self, mock_llm, mock_get, mock_save, mock_ctx):
        event = make_event("IntentRequest", intent_name="AskClaudeIntent", query="test question")
        lambda_function.lambda_handler(event, None)
        saved_history = mock_save.call_args[0][1]
        self.assertEqual(len(saved_history), 2)
        self.assertEqual(saved_history[0]["role"], "user")
        self.assertEqual(saved_history[0]["content"], "test question")
        self.assertEqual(saved_history[1]["role"], "assistant")
        self.assertEqual(saved_history[1]["content"], "Sure.")

    @patch("lambda_function.get_user_context", return_value="I am a software engineer")
    @patch("lambda_function.save_history")
    @patch("lambda_function.get_history", return_value=[])
    @patch("lambda_function.ask_llm", return_value="Tailored answer.")
    def test_user_context_passed_to_llm(self, mock_llm, mock_get, mock_save, mock_ctx):
        event = make_event("IntentRequest", intent_name="AskClaudeIntent", query="help me debug")
        lambda_function.lambda_handler(event, None)
        _, _, context_arg = mock_llm.call_args[0]
        self.assertEqual(context_arg, "I am a software engineer")

    @patch("lambda_function.get_user_context", return_value="")
    @patch("lambda_function.save_history")
    @patch("lambda_function.get_history", return_value=[])
    @patch("lambda_function.ask_llm", return_value="Answer.")
    def test_correct_user_id_used(self, mock_llm, mock_get, mock_save, mock_ctx):
        event = make_event("IntentRequest", intent_name="AskClaudeIntent", query="q", user_id="user-xyz")
        lambda_function.lambda_handler(event, None)
        mock_get.assert_called_once_with("user-xyz")
        mock_save.assert_called_once_with("user-xyz", unittest.mock.ANY)

    @patch("lambda_function.get_user_context", return_value="")
    @patch("lambda_function.save_history")
    @patch("lambda_function.get_history", return_value=[])
    @patch("lambda_function.ask_llm", return_value="OK.")
    def test_session_stays_open_after_answer(self, mock_llm, mock_get, mock_save, mock_ctx):
        event = make_event("IntentRequest", intent_name="AskClaudeIntent", query="anything")
        r = lambda_function.lambda_handler(event, None)
        self.assertFalse(r["response"]["shouldEndSession"])

    @patch("lambda_function.get_user_context", return_value="")
    @patch("lambda_function.save_history")
    @patch("lambda_function.get_history", return_value=[
        {"role": "user", "content": "msg"},
        {"role": "assistant", "content": "reply"},
    ] * 10)
    @patch("lambda_function.ask_llm", return_value="OK.")
    def test_long_history_still_works(self, mock_llm, mock_get, mock_save, mock_ctx):
        event = make_event("IntentRequest", intent_name="AskClaudeIntent", query="latest question")
        r = lambda_function.lambda_handler(event, None)
        self.assertEqual(get_speech(r), "OK.")


# ══════════════════════════════════════════════════════════════════════════════
# 4. YesIntent / NoIntent
# ══════════════════════════════════════════════════════════════════════════════

class TestYesNoIntent(unittest.TestCase):

    @patch("lambda_function.get_user_context", return_value="")
    @patch("lambda_function.save_history")
    @patch("lambda_function.get_history", return_value=[])
    @patch("lambda_function.ask_llm", return_value="Great, let me continue.")
    def test_yes_keeps_session_open(self, mock_llm, mock_get, mock_save, mock_ctx):
        event = make_event("IntentRequest", intent_name="AMAZON.YesIntent")
        r = lambda_function.lambda_handler(event, None)
        self.assertFalse(r["response"]["shouldEndSession"])

    @patch("lambda_function.get_user_context", return_value="")
    @patch("lambda_function.save_history")
    @patch("lambda_function.get_history", return_value=[])
    @patch("lambda_function.ask_llm", return_value="Alright, goodbye.")
    def test_no_ends_session(self, mock_llm, mock_get, mock_save, mock_ctx):
        event = make_event("IntentRequest", intent_name="AMAZON.NoIntent")
        r = lambda_function.lambda_handler(event, None)
        self.assertTrue(r["response"]["shouldEndSession"])

    @patch("lambda_function.get_user_context", return_value="")
    @patch("lambda_function.save_history")
    @patch("lambda_function.get_history", return_value=[])
    @patch("lambda_function.ask_llm", return_value="Sure.")
    def test_yes_sends_word_yes_to_llm(self, mock_llm, mock_get, mock_save, mock_ctx):
        event = make_event("IntentRequest", intent_name="AMAZON.YesIntent")
        lambda_function.lambda_handler(event, None)
        self.assertEqual(mock_llm.call_args[0][0], "yes")

    @patch("lambda_function.get_user_context", return_value="")
    @patch("lambda_function.save_history")
    @patch("lambda_function.get_history", return_value=[])
    @patch("lambda_function.ask_llm", return_value="Goodbye.")
    def test_no_sends_word_no_to_llm(self, mock_llm, mock_get, mock_save, mock_ctx):
        event = make_event("IntentRequest", intent_name="AMAZON.NoIntent")
        lambda_function.lambda_handler(event, None)
        self.assertEqual(mock_llm.call_args[0][0], "no")

    @patch("lambda_function.get_user_context", return_value="")
    @patch("lambda_function.save_history")
    @patch("lambda_function.get_history", return_value=[])
    @patch("lambda_function.ask_llm", side_effect=Exception("timeout"))
    def test_llm_failure_returns_error(self, mock_llm, mock_get, mock_save, mock_ctx):
        event = make_event("IntentRequest", intent_name="AMAZON.YesIntent")
        r = lambda_function.lambda_handler(event, None)
        self.assertIn("trouble", get_speech(r))

    @patch("lambda_function.get_user_context", return_value="")
    @patch("lambda_function.save_history")
    @patch("lambda_function.get_history", return_value=[])
    @patch("lambda_function.ask_llm", return_value="OK.")
    def test_history_saved_on_yes(self, mock_llm, mock_get, mock_save, mock_ctx):
        event = make_event("IntentRequest", intent_name="AMAZON.YesIntent")
        lambda_function.lambda_handler(event, None)
        mock_save.assert_called_once()
        saved = mock_save.call_args[0][1]
        self.assertEqual(saved[-2]["content"], "yes")
        self.assertEqual(saved[-1]["content"], "OK.")


# ══════════════════════════════════════════════════════════════════════════════
# 5. SetContextIntent
# ══════════════════════════════════════════════════════════════════════════════

class TestSetContextIntent(unittest.TestCase):

    @patch("lambda_function.save_user_context")
    def test_saves_context(self, mock_save):
        event = make_event("IntentRequest", intent_name="SetContextIntent", context_value="I'm a nurse")
        r = lambda_function.lambda_handler(event, None)
        mock_save.assert_called_once_with("test-user-id", "I'm a nurse")
        self.assertIn("Got it", get_speech(r))

    @patch("lambda_function.save_user_context")
    def test_empty_context_slot_returns_prompt(self, mock_save):
        event = make_event("IntentRequest", intent_name="SetContextIntent")
        r = lambda_function.lambda_handler(event, None)
        mock_save.assert_not_called()
        self.assertIn("didn't catch", get_speech(r))

    @patch("lambda_function.save_user_context")
    def test_saves_to_correct_user(self, mock_save):
        event = make_event("IntentRequest", intent_name="SetContextIntent", context_value="context", user_id="user-abc")
        lambda_function.lambda_handler(event, None)
        mock_save.assert_called_once_with("user-abc", "context")

    @patch("lambda_function.save_user_context")
    def test_session_stays_open(self, mock_save):
        event = make_event("IntentRequest", intent_name="SetContextIntent", context_value="I like jazz")
        r = lambda_function.lambda_handler(event, None)
        self.assertFalse(r["response"]["shouldEndSession"])


# ══════════════════════════════════════════════════════════════════════════════
# 6. ClearContextIntent
# ══════════════════════════════════════════════════════════════════════════════

class TestClearContextIntent(unittest.TestCase):

    @patch("lambda_function.clear_user_context")
    def test_clears_context(self, mock_clear):
        event = make_event("IntentRequest", intent_name="ClearContextIntent")
        r = lambda_function.lambda_handler(event, None)
        mock_clear.assert_called_once_with("test-user-id")
        self.assertIn("cleared", get_speech(r))

    @patch("lambda_function.clear_user_context")
    def test_clears_correct_user(self, mock_clear):
        event = make_event("IntentRequest", intent_name="ClearContextIntent", user_id="user-xyz")
        lambda_function.lambda_handler(event, None)
        mock_clear.assert_called_once_with("user-xyz")

    @patch("lambda_function.clear_user_context")
    def test_session_stays_open(self, mock_clear):
        event = make_event("IntentRequest", intent_name="ClearContextIntent")
        r = lambda_function.lambda_handler(event, None)
        self.assertFalse(r["response"]["shouldEndSession"])


# ══════════════════════════════════════════════════════════════════════════════
# 7. HelpIntent
# ══════════════════════════════════════════════════════════════════════════════

class TestHelpIntent(unittest.TestCase):

    def test_returns_help_text(self):
        event = make_event("IntentRequest", intent_name="AMAZON.HelpIntent")
        r = lambda_function.lambda_handler(event, None)
        self.assertTrue(len(get_speech(r)) > 0)

    def test_session_stays_open(self):
        event = make_event("IntentRequest", intent_name="AMAZON.HelpIntent")
        r = lambda_function.lambda_handler(event, None)
        self.assertFalse(r["response"]["shouldEndSession"])


# ══════════════════════════════════════════════════════════════════════════════
# 8. StopIntent / CancelIntent
# ══════════════════════════════════════════════════════════════════════════════

class TestStopIntent(unittest.TestCase):

    def test_stop_ends_session(self):
        event = make_event("IntentRequest", intent_name="AMAZON.StopIntent")
        r = lambda_function.lambda_handler(event, None)
        self.assertTrue(r["response"]["shouldEndSession"])

    def test_stop_says_goodbye(self):
        event = make_event("IntentRequest", intent_name="AMAZON.StopIntent")
        r = lambda_function.lambda_handler(event, None)
        self.assertIn("Goodbye", get_speech(r))

    def test_cancel_ends_session(self):
        event = make_event("IntentRequest", intent_name="AMAZON.CancelIntent")
        r = lambda_function.lambda_handler(event, None)
        self.assertTrue(r["response"]["shouldEndSession"])

    def test_cancel_says_goodbye(self):
        event = make_event("IntentRequest", intent_name="AMAZON.CancelIntent")
        r = lambda_function.lambda_handler(event, None)
        self.assertIn("Goodbye", get_speech(r))


# ══════════════════════════════════════════════════════════════════════════════
# 9. FallbackIntent & unknown intents
# ══════════════════════════════════════════════════════════════════════════════

FALLBACK_RESPONSES = [
    "How are you feeling about that?",
    "Tell me more, I'm listening.",
    "How did that make you feel?",
    "What's on your mind?",
    "I'm here. What would you like to talk about?",
    "How are you feeling right now?",
    "Want to talk about it?",
]


class TestFallbackIntent(unittest.TestCase):

    def test_fallback_intent_returns_valid_response(self):
        event = make_event("IntentRequest", intent_name="AMAZON.FallbackIntent")
        r = lambda_function.lambda_handler(event, None)
        self.assertIn(get_speech(r), FALLBACK_RESPONSES)

    def test_unknown_intent_returns_fallback(self):
        event = make_event("IntentRequest", intent_name="SomeRandomIntent")
        r = lambda_function.lambda_handler(event, None)
        self.assertIn(get_speech(r), FALLBACK_RESPONSES)

    def test_fallback_session_stays_open(self):
        event = make_event("IntentRequest", intent_name="AMAZON.FallbackIntent")
        r = lambda_function.lambda_handler(event, None)
        self.assertFalse(r["response"]["shouldEndSession"])

    def test_fallback_is_randomised(self):
        """Over 50 calls at least 2 distinct responses should appear."""
        event = make_event("IntentRequest", intent_name="AMAZON.FallbackIntent")
        speeches = set()
        for _ in range(50):
            r = lambda_function.lambda_handler(event, None)
            speeches.add(get_speech(r))
        self.assertGreater(len(speeches), 1)


# ══════════════════════════════════════════════════════════════════════════════
# 10. SessionEndedRequest
# ══════════════════════════════════════════════════════════════════════════════

class TestSessionEndedRequest(unittest.TestCase):

    def test_ends_session(self):
        event = make_event("SessionEndedRequest")
        r = lambda_function.lambda_handler(event, None)
        self.assertTrue(r["response"]["shouldEndSession"])

    def test_user_initiated_reason(self):
        event = make_event("SessionEndedRequest", session_reason="USER_INITIATED")
        r = lambda_function.lambda_handler(event, None)
        self.assertTrue(r["response"]["shouldEndSession"])

    def test_error_reason(self):
        event = make_event("SessionEndedRequest", session_reason="ERROR")
        r = lambda_function.lambda_handler(event, None)
        self.assertTrue(r["response"]["shouldEndSession"])

    def test_exceeded_max_reprompts_reason(self):
        event = make_event("SessionEndedRequest", session_reason="EXCEEDED_MAX_REPROMPTS")
        r = lambda_function.lambda_handler(event, None)
        self.assertTrue(r["response"]["shouldEndSession"])


# ══════════════════════════════════════════════════════════════════════════════
# 11. Unknown request type
# ══════════════════════════════════════════════════════════════════════════════

class TestUnknownRequestType(unittest.TestCase):

    def test_unknown_type_returns_response(self):
        event = make_event("LaunchRequest")
        event["request"]["type"] = "WeirdRequestType"
        r = lambda_function.lambda_handler(event, None)
        self.assertIn("version", r)


# ══════════════════════════════════════════════════════════════════════════════
# 12. DynamoDB layer
# ══════════════════════════════════════════════════════════════════════════════

class TestDynamo(unittest.TestCase):

    def setUp(self):
        # Patch boto3 before importing dynamo so we never hit AWS
        self.boto3_patcher = patch("boto3.resource")
        self.mock_boto3 = self.boto3_patcher.start()

        self.mock_table = MagicMock()
        self.mock_boto3.return_value.Table.return_value = self.mock_table

        # Force dynamo module to re-initialise its cached table reference
        import dynamo
        dynamo._table = None
        self.dynamo = dynamo

    def tearDown(self):
        self.boto3_patcher.stop()
        import dynamo
        dynamo._table = None

    # ── get_history ────────────────────────────────────────────────

    def test_get_history_returns_list_when_item_exists(self):
        history = [{"role": "user", "content": "hi"}]
        self.mock_table.get_item.return_value = {"Item": {"userId": "u1", "history": history}}
        result = self.dynamo.get_history("u1")
        self.assertEqual(result, history)

    def test_get_history_returns_empty_list_when_no_item(self):
        self.mock_table.get_item.return_value = {}
        result = self.dynamo.get_history("u1")
        self.assertEqual(result, [])

    def test_get_history_returns_empty_list_when_no_history_key(self):
        self.mock_table.get_item.return_value = {"Item": {"userId": "u1"}}
        result = self.dynamo.get_history("u1")
        self.assertEqual(result, [])

    def test_get_history_calls_correct_user_id(self):
        self.mock_table.get_item.return_value = {}
        self.dynamo.get_history("my-special-user")
        self.mock_table.get_item.assert_called_once_with(Key={"userId": "my-special-user"})

    # ── save_history ───────────────────────────────────────────────

    def test_save_history_calls_update_item(self):
        history = [{"role": "user", "content": "hello"}]
        self.dynamo.save_history("u1", history)
        self.mock_table.update_item.assert_called_once()

    def test_save_history_uses_correct_key(self):
        self.dynamo.save_history("user-abc", [])
        call_kwargs = self.mock_table.update_item.call_args[1]
        self.assertEqual(call_kwargs["Key"], {"userId": "user-abc"})

    def test_save_history_passes_history_in_expression(self):
        history = [{"role": "assistant", "content": "Hi!"}]
        self.dynamo.save_history("u1", history)
        call_kwargs = self.mock_table.update_item.call_args[1]
        self.assertEqual(call_kwargs["ExpressionAttributeValues"][":h"], history)

    def test_save_empty_history(self):
        self.dynamo.save_history("u1", [])
        call_kwargs = self.mock_table.update_item.call_args[1]
        self.assertEqual(call_kwargs["ExpressionAttributeValues"][":h"], [])

    # ── get_user_context ───────────────────────────────────────────

    def test_get_user_context_returns_value_when_exists(self):
        self.mock_table.get_item.return_value = {"Item": {"userId": "u1", "userContext": "I am a chef"}}
        result = self.dynamo.get_user_context("u1")
        self.assertEqual(result, "I am a chef")

    def test_get_user_context_returns_empty_string_when_no_item(self):
        self.mock_table.get_item.return_value = {}
        result = self.dynamo.get_user_context("u1")
        self.assertEqual(result, "")

    def test_get_user_context_returns_empty_string_when_no_key(self):
        self.mock_table.get_item.return_value = {"Item": {"userId": "u1"}}
        result = self.dynamo.get_user_context("u1")
        self.assertEqual(result, "")

    # ── save_user_context ──────────────────────────────────────────

    def test_save_user_context_calls_update_item(self):
        self.dynamo.save_user_context("u1", "I am a pilot")
        self.mock_table.update_item.assert_called_once()

    def test_save_user_context_uses_correct_key(self):
        self.dynamo.save_user_context("u-xyz", "context")
        call_kwargs = self.mock_table.update_item.call_args[1]
        self.assertEqual(call_kwargs["Key"], {"userId": "u-xyz"})

    def test_save_user_context_passes_value(self):
        self.dynamo.save_user_context("u1", "I like jazz")
        call_kwargs = self.mock_table.update_item.call_args[1]
        self.assertEqual(call_kwargs["ExpressionAttributeValues"][":c"], "I like jazz")

    # ── clear_user_context ─────────────────────────────────────────

    def test_clear_user_context_calls_update_item(self):
        self.dynamo.clear_user_context("u1")
        self.mock_table.update_item.assert_called_once()

    def test_clear_user_context_uses_remove_expression(self):
        self.dynamo.clear_user_context("u1")
        call_kwargs = self.mock_table.update_item.call_args[1]
        self.assertIn("REMOVE", call_kwargs["UpdateExpression"])

    def test_clear_user_context_uses_correct_key(self):
        self.dynamo.clear_user_context("user-abc")
        call_kwargs = self.mock_table.update_item.call_args[1]
        self.assertEqual(call_kwargs["Key"], {"userId": "user-abc"})

    # ── table name resolution ──────────────────────────────────────

    def test_custom_table_name_from_env(self):
        import dynamo
        dynamo._table = None
        with patch.dict(os.environ, {"DYNAMODB_TABLE": "MyCustomTable"}):
            dynamo._table = None
            self.mock_table.get_item.return_value = {}
            dynamo.get_history("u1")
            self.mock_boto3.return_value.Table.assert_called_with("MyCustomTable")

    def test_default_table_name(self):
        import dynamo
        dynamo._table = None
        env = {k: v for k, v in os.environ.items() if k != "DYNAMODB_TABLE"}
        with patch.dict(os.environ, env, clear=True):
            dynamo._table = None
            self.mock_table.get_item.return_value = {}
            dynamo.get_history("u1")
            self.mock_boto3.return_value.Table.assert_called_with("AlexaConversationHistory")


# ══════════════════════════════════════════════════════════════════════════════
# 13. Groq provider
# ══════════════════════════════════════════════════════════════════════════════

class TestGroqProvider(unittest.TestCase):

    def setUp(self):
        # Re-import fresh with the test API key in place
        import groq_provider
        self.provider = groq_provider

    def _make_groq_response(self, content="Hello from Groq."):
        return {"choices": [{"message": {"content": content}}]}

    @patch("urllib.request.urlopen")
    def test_success_returns_content(self, mock_open):
        mock_open.return_value = make_urlopen_mock(self._make_groq_response("Groq says hi."))
        result = self.provider.ask_llm("say hi", [], "")
        self.assertEqual(result, "Groq says hi.")

    @patch("urllib.request.urlopen")
    def test_sends_user_message(self, mock_open):
        mock_open.return_value = make_urlopen_mock(self._make_groq_response())
        self.provider.ask_llm("what is AI", [], "")
        body = json.loads(mock_open.call_args[0][0].data.decode())
        user_messages = [m for m in body["messages"] if m["role"] == "user"]
        self.assertEqual(user_messages[-1]["content"], "what is AI")

    @patch("urllib.request.urlopen")
    def test_sends_conversation_history(self, mock_open):
        mock_open.return_value = make_urlopen_mock(self._make_groq_response())
        history = [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
        ]
        self.provider.ask_llm("second question", history, "")
        body = json.loads(mock_open.call_args[0][0].data.decode())
        # system + 2 history + 1 new user = 4 messages
        self.assertEqual(len(body["messages"]), 4)

    @patch("urllib.request.urlopen")
    def test_system_prompt_included(self, mock_open):
        mock_open.return_value = make_urlopen_mock(self._make_groq_response())
        self.provider.ask_llm("hello", [], "")
        body = json.loads(mock_open.call_args[0][0].data.decode())
        self.assertEqual(body["messages"][0]["role"], "system")

    @patch("urllib.request.urlopen")
    def test_user_context_appended_to_system_prompt(self, mock_open):
        mock_open.return_value = make_urlopen_mock(self._make_groq_response())
        self.provider.ask_llm("hello", [], "I am a teacher")
        body = json.loads(mock_open.call_args[0][0].data.decode())
        self.assertIn("I am a teacher", body["messages"][0]["content"])

    @patch("urllib.request.urlopen")
    def test_empty_context_not_appended(self, mock_open):
        mock_open.return_value = make_urlopen_mock(self._make_groq_response())
        self.provider.ask_llm("hello", [], "")
        body = json.loads(mock_open.call_args[0][0].data.decode())
        self.assertNotIn("personal context", body["messages"][0]["content"])

    @patch("urllib.request.urlopen")
    def test_none_history_defaults_to_empty(self, mock_open):
        mock_open.return_value = make_urlopen_mock(self._make_groq_response())
        result = self.provider.ask_llm("hi", None, None)
        self.assertEqual(result, "Hello from Groq.")

    @patch("urllib.request.urlopen")
    def test_http_error_raises(self, mock_open):
        import urllib.error
        mock_open.side_effect = urllib.error.HTTPError(
            url="https://api.groq.com",
            code=429,
            msg="Too Many Requests",
            hdrs={},
            fp=io.BytesIO(b'{"error":"rate limit"}'),
        )
        with self.assertRaises(urllib.error.HTTPError):
            self.provider.ask_llm("hi", [], "")

    def test_missing_api_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                self.provider.ask_llm("hi", [], "")

    @patch("urllib.request.urlopen")
    def test_authorization_header_sent(self, mock_open):
        mock_open.return_value = make_urlopen_mock(self._make_groq_response())
        self.provider.ask_llm("hi", [], "")
        req = mock_open.call_args[0][0]
        self.assertIn("Authorization", req.headers)
        self.assertTrue(req.headers["Authorization"].startswith("Bearer "))

    @patch("urllib.request.urlopen")
    def test_post_method_used(self, mock_open):
        mock_open.return_value = make_urlopen_mock(self._make_groq_response())
        self.provider.ask_llm("hi", [], "")
        req = mock_open.call_args[0][0]
        self.assertEqual(req.get_method(), "POST")

    @patch("urllib.request.urlopen")
    def test_max_tokens_in_payload(self, mock_open):
        mock_open.return_value = make_urlopen_mock(self._make_groq_response())
        self.provider.ask_llm("hi", [], "")
        body = json.loads(mock_open.call_args[0][0].data.decode())
        self.assertIn("max_tokens", body)


# ══════════════════════════════════════════════════════════════════════════════
# 14. Gemini provider
# ══════════════════════════════════════════════════════════════════════════════

class TestGeminiProvider(unittest.TestCase):

    def setUp(self):
        import gemini_provider
        self.provider = gemini_provider

    def _make_gemini_response(self, content="Hello from Gemini."):
        return {
            "candidates": [
                {"content": {"parts": [{"text": content}]}}
            ]
        }

    @patch("urllib.request.urlopen")
    def test_success_returns_content(self, mock_open):
        mock_open.return_value = make_urlopen_mock(self._make_gemini_response("Gemini answer."))
        result = self.provider.ask_llm("question", [], "")
        self.assertEqual(result, "Gemini answer.")

    @patch("urllib.request.urlopen")
    def test_history_role_user_maps_correctly(self, mock_open):
        mock_open.return_value = make_urlopen_mock(self._make_gemini_response())
        history = [{"role": "user", "content": "hi"}]
        self.provider.ask_llm("follow up", history, "")
        body = json.loads(mock_open.call_args[0][0].data.decode())
        self.assertEqual(body["contents"][0]["role"], "user")

    @patch("urllib.request.urlopen")
    def test_history_role_assistant_maps_to_model(self, mock_open):
        mock_open.return_value = make_urlopen_mock(self._make_gemini_response())
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        self.provider.ask_llm("next", history, "")
        body = json.loads(mock_open.call_args[0][0].data.decode())
        self.assertEqual(body["contents"][1]["role"], "model")

    @patch("urllib.request.urlopen")
    def test_user_message_appended_as_last_content(self, mock_open):
        mock_open.return_value = make_urlopen_mock(self._make_gemini_response())
        self.provider.ask_llm("final question", [], "")
        body = json.loads(mock_open.call_args[0][0].data.decode())
        last = body["contents"][-1]
        self.assertEqual(last["role"], "user")
        self.assertEqual(last["parts"][0]["text"], "final question")

    @patch("urllib.request.urlopen")
    def test_system_instruction_included(self, mock_open):
        mock_open.return_value = make_urlopen_mock(self._make_gemini_response())
        self.provider.ask_llm("hi", [], "")
        body = json.loads(mock_open.call_args[0][0].data.decode())
        self.assertIn("systemInstruction", body)

    @patch("urllib.request.urlopen")
    def test_user_context_in_system_instruction(self, mock_open):
        mock_open.return_value = make_urlopen_mock(self._make_gemini_response())
        self.provider.ask_llm("hi", [], "I am a doctor")
        body = json.loads(mock_open.call_args[0][0].data.decode())
        system_text = body["systemInstruction"]["parts"][0]["text"]
        self.assertIn("I am a doctor", system_text)

    @patch("urllib.request.urlopen")
    def test_none_history_defaults_to_empty(self, mock_open):
        mock_open.return_value = make_urlopen_mock(self._make_gemini_response("OK."))
        result = self.provider.ask_llm("hi", None, None)
        self.assertEqual(result, "OK.")

    @patch("urllib.request.urlopen")
    def test_http_error_raises(self, mock_open):
        import urllib.error
        mock_open.side_effect = urllib.error.HTTPError(
            url="https://generativelanguage.googleapis.com",
            code=403,
            msg="Forbidden",
            hdrs={},
            fp=io.BytesIO(b'{"error":"forbidden"}'),
        )
        with self.assertRaises(urllib.error.HTTPError):
            self.provider.ask_llm("hi", [], "")

    def test_missing_api_key_raises(self):
        env = {k: v for k, v in os.environ.items() if k != "GEMINI_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError):
                self.provider.ask_llm("hi", [], "")

    @patch("urllib.request.urlopen")
    def test_api_key_in_url_not_header(self, mock_open):
        mock_open.return_value = make_urlopen_mock(self._make_gemini_response())
        self.provider.ask_llm("hi", [], "")
        req = mock_open.call_args[0][0]
        self.assertIn("key=", req.full_url)

    @patch("urllib.request.urlopen")
    def test_generation_config_included(self, mock_open):
        mock_open.return_value = make_urlopen_mock(self._make_gemini_response())
        self.provider.ask_llm("hi", [], "")
        body = json.loads(mock_open.call_args[0][0].data.decode())
        self.assertIn("generationConfig", body)


# ══════════════════════════════════════════════════════════════════════════════
# 15. OpenRouter provider
# ══════════════════════════════════════════════════════════════════════════════

class TestOpenRouterProvider(unittest.TestCase):

    def setUp(self):
        import openrouter_provider
        self.provider = openrouter_provider

    def _make_or_response(self, content="Hello from OpenRouter."):
        return {"choices": [{"message": {"content": content}}]}

    @patch("urllib.request.urlopen")
    def test_success_returns_content(self, mock_open):
        mock_open.return_value = make_urlopen_mock(self._make_or_response("OR answer."))
        result = self.provider.ask_llm("question", [], "")
        self.assertEqual(result, "OR answer.")

    @patch("urllib.request.urlopen")
    def test_sends_user_message(self, mock_open):
        mock_open.return_value = make_urlopen_mock(self._make_or_response())
        self.provider.ask_llm("test query", [], "")
        body = json.loads(mock_open.call_args[0][0].data.decode())
        user_msgs = [m for m in body["messages"] if m["role"] == "user"]
        self.assertEqual(user_msgs[-1]["content"], "test query")

    @patch("urllib.request.urlopen")
    def test_system_prompt_included(self, mock_open):
        mock_open.return_value = make_urlopen_mock(self._make_or_response())
        self.provider.ask_llm("hi", [], "")
        body = json.loads(mock_open.call_args[0][0].data.decode())
        self.assertEqual(body["messages"][0]["role"], "system")

    @patch("urllib.request.urlopen")
    def test_user_context_in_system_prompt(self, mock_open):
        mock_open.return_value = make_urlopen_mock(self._make_or_response())
        self.provider.ask_llm("hi", [], "I am an architect")
        body = json.loads(mock_open.call_args[0][0].data.decode())
        self.assertIn("I am an architect", body["messages"][0]["content"])

    @patch("urllib.request.urlopen")
    def test_sends_conversation_history(self, mock_open):
        mock_open.return_value = make_urlopen_mock(self._make_or_response())
        history = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
        self.provider.ask_llm("follow up", history, "")
        body = json.loads(mock_open.call_args[0][0].data.decode())
        # system + 2 history + 1 user = 4
        self.assertEqual(len(body["messages"]), 4)

    @patch("urllib.request.urlopen")
    def test_authorization_header_sent(self, mock_open):
        mock_open.return_value = make_urlopen_mock(self._make_or_response())
        self.provider.ask_llm("hi", [], "")
        req = mock_open.call_args[0][0]
        self.assertTrue(req.headers["Authorization"].startswith("Bearer "))

    @patch("urllib.request.urlopen")
    def test_site_headers_present(self, mock_open):
        mock_open.return_value = make_urlopen_mock(self._make_or_response())
        self.provider.ask_llm("hi", [], "")
        req = mock_open.call_args[0][0]
        # OpenRouter requires HTTP-Referer and X-Title
        self.assertIn("Http-referer", req.headers)
        self.assertIn("X-title", req.headers)

    @patch("urllib.request.urlopen")
    def test_http_error_raises(self, mock_open):
        import urllib.error
        mock_open.side_effect = urllib.error.HTTPError(
            url="https://openrouter.ai",
            code=500,
            msg="Internal Server Error",
            hdrs={},
            fp=io.BytesIO(b'{"error":"server error"}'),
        )
        with self.assertRaises(urllib.error.HTTPError):
            self.provider.ask_llm("hi", [], "")

    def test_missing_api_key_raises(self):
        env = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError):
                self.provider.ask_llm("hi", [], "")

    @patch("urllib.request.urlopen")
    def test_none_history_defaults_to_empty(self, mock_open):
        mock_open.return_value = make_urlopen_mock(self._make_or_response("OK."))
        result = self.provider.ask_llm("hi", None, None)
        self.assertEqual(result, "OK.")

    @patch("urllib.request.urlopen")
    def test_post_method_used(self, mock_open):
        mock_open.return_value = make_urlopen_mock(self._make_or_response())
        self.provider.ask_llm("hi", [], "")
        req = mock_open.call_args[0][0]
        self.assertEqual(req.get_method(), "POST")


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
