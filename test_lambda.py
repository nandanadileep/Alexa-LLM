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

    @patch("lambda_function.clear_pending_chunks")
    @patch("lambda_function.get_user_facts", return_value={})
    @patch("lambda_function.save_history")
    @patch("lambda_function.get_history", return_value=[])
    @patch("lambda_function.call_llm", return_value="Stoicism is a philosophy of resilience.")
    def test_success_with_query_slot(self, mock_llm, mock_get, mock_save, mock_facts, *_):
        event = make_event("IntentRequest", intent_name="AskClaudeIntent", query="what is stoicism")
        r = lambda_function.lambda_handler(event, None)
        self.assertEqual(get_speech(r), "Stoicism is a philosophy of resilience.")
        mock_llm.assert_called_once_with("what is stoicism", [], "")

    @patch("lambda_function.clear_pending_chunks")
    @patch("lambda_function.get_user_facts", return_value={})
    @patch("lambda_function.save_history")
    @patch("lambda_function.get_history", return_value=[])
    @patch("lambda_function.call_llm", return_value="Stoicism is about self-control.")
    def test_success_with_question_slot(self, mock_llm, mock_get, mock_save, mock_facts, *_):
        event = make_event("IntentRequest", intent_name="AskClaudeIntent", question="tell me about stoicism")
        r = lambda_function.lambda_handler(event, None)
        self.assertEqual(get_speech(r), "Stoicism is about self-control.")
        mock_llm.assert_called_once_with("tell me about stoicism", [], "")

    @patch("lambda_function.get_user_facts", return_value={})
    @patch("lambda_function.save_history")
    @patch("lambda_function.get_history", return_value=[])
    def test_no_query_returns_prompt(self, mock_get, mock_save, *_):
        event = make_event("IntentRequest", intent_name="AskClaudeIntent")
        r = lambda_function.lambda_handler(event, None)
        self.assertIn("didn't catch", get_speech(r))
        mock_save.assert_not_called()

    @patch("lambda_function.get_user_facts", return_value={})
    @patch("lambda_function.save_history")
    @patch("lambda_function.get_history", return_value=[])
    @patch("lambda_function.call_llm", side_effect=Exception("API timeout"))
    def test_llm_failure_returns_error_message(self, *_):
        event = make_event("IntentRequest", intent_name="AskClaudeIntent", query="hello")
        r = lambda_function.lambda_handler(event, None)
        self.assertIn("trouble", get_speech(r))

    @patch("lambda_function.get_user_facts", return_value={})
    @patch("lambda_function.save_history")
    @patch("lambda_function.get_history", return_value=[])
    @patch("lambda_function.call_llm", side_effect=Exception("API timeout"))
    def test_llm_failure_does_not_save_history(self, mock_llm, mock_get, mock_save, *_):
        event = make_event("IntentRequest", intent_name="AskClaudeIntent", query="hello")
        lambda_function.lambda_handler(event, None)
        mock_save.assert_not_called()

    @patch("lambda_function.clear_pending_chunks")
    @patch("lambda_function.get_user_facts", return_value={})
    @patch("lambda_function.save_history")
    @patch("lambda_function.get_history", return_value=[
        {"role": "user", "content": "what is stoicism"},
        {"role": "assistant", "content": "Stoicism is a philosophy."},
    ])
    @patch("lambda_function.call_llm", return_value="Epictetus was a Stoic philosopher.")
    def test_history_passed_to_llm(self, mock_llm, mock_get, mock_save, *_):
        event = make_event("IntentRequest", intent_name="AskClaudeIntent", query="who is epictetus")
        lambda_function.lambda_handler(event, None)
        _, history_arg, _ = mock_llm.call_args[0]
        self.assertEqual(len(history_arg), 2)
        self.assertEqual(history_arg[0]["role"], "user")

    @patch("lambda_function.clear_pending_chunks")
    @patch("lambda_function.get_user_facts", return_value={})
    @patch("lambda_function.save_history")
    @patch("lambda_function.get_history", return_value=[])
    @patch("lambda_function.call_llm", return_value="Sure.")
    def test_history_updated_with_new_turn(self, mock_llm, mock_get, mock_save, *_):
        event = make_event("IntentRequest", intent_name="AskClaudeIntent", query="test question")
        lambda_function.lambda_handler(event, None)
        saved_history = mock_save.call_args[0][1]
        self.assertEqual(len(saved_history), 2)
        self.assertEqual(saved_history[0]["role"], "user")
        self.assertEqual(saved_history[0]["content"], "test question")
        self.assertEqual(saved_history[1]["role"], "assistant")
        self.assertEqual(saved_history[1]["content"], "Sure.")

    @patch("lambda_function.clear_pending_chunks")
    @patch("lambda_function.get_user_facts", return_value={"job": "software engineer"})
    @patch("lambda_function.save_history")
    @patch("lambda_function.get_history", return_value=[])
    @patch("lambda_function.call_llm", return_value="Tailored answer.")
    def test_user_facts_formatted_and_passed_to_llm(self, mock_llm, *_):
        event = make_event("IntentRequest", intent_name="AskClaudeIntent", query="help me debug")
        lambda_function.lambda_handler(event, None)
        _, _, context_arg = mock_llm.call_args[0]
        self.assertEqual(context_arg, "job: software engineer")

    @patch("lambda_function.clear_pending_chunks")
    @patch("lambda_function.get_user_facts", return_value={})
    @patch("lambda_function.save_history")
    @patch("lambda_function.get_history", return_value=[])
    @patch("lambda_function.call_llm", return_value="Answer.")
    def test_correct_user_id_used(self, mock_llm, mock_get, mock_save, *_):
        event = make_event("IntentRequest", intent_name="AskClaudeIntent", query="q", user_id="user-xyz")
        lambda_function.lambda_handler(event, None)
        mock_get.assert_called_once_with("user-xyz")
        mock_save.assert_called_once_with("user-xyz", unittest.mock.ANY)

    @patch("lambda_function.clear_pending_chunks")
    @patch("lambda_function.get_user_facts", return_value={})
    @patch("lambda_function.save_history")
    @patch("lambda_function.get_history", return_value=[])
    @patch("lambda_function.call_llm", return_value="OK.")
    def test_session_stays_open_after_answer(self, *_):
        event = make_event("IntentRequest", intent_name="AskClaudeIntent", query="anything")
        r = lambda_function.lambda_handler(event, None)
        self.assertFalse(r["response"]["shouldEndSession"])

    @patch("lambda_function.clear_pending_chunks")
    @patch("lambda_function.get_user_facts", return_value={})
    @patch("lambda_function.save_history")
    @patch("lambda_function.get_history", return_value=[
        {"role": "user", "content": "msg"},
        {"role": "assistant", "content": "reply"},
    ] * 10)
    @patch("lambda_function.call_llm", return_value="OK.")
    def test_long_history_still_works(self, *_):
        event = make_event("IntentRequest", intent_name="AskClaudeIntent", query="latest question")
        r = lambda_function.lambda_handler(event, None)
        self.assertEqual(get_speech(r), "OK.")


# ══════════════════════════════════════════════════════════════════════════════
# 4. YesIntent / NoIntent
# ══════════════════════════════════════════════════════════════════════════════

class TestYesNoIntent(unittest.TestCase):
    # YesIntent → handle_continue_intent (delivers next pending chunk)
    # NoIntent  → clears pending chunks, keeps session open

    @patch("lambda_function.save_pending_chunks")
    @patch("lambda_function.get_pending_chunks", return_value=["Part two.", "Part three."])
    def test_yes_delivers_next_chunk(self, *_):
        event = make_event("IntentRequest", intent_name="AMAZON.YesIntent")
        r = lambda_function.lambda_handler(event, None)
        self.assertIn("Part two", get_speech(r))

    @patch("lambda_function.save_pending_chunks")
    @patch("lambda_function.get_pending_chunks", return_value=["Part two.", "Part three."])
    def test_yes_prompts_to_continue_when_more_chunks_remain(self, *_):
        event = make_event("IntentRequest", intent_name="AMAZON.YesIntent")
        r = lambda_function.lambda_handler(event, None)
        self.assertIn("continue", get_speech(r).lower())

    @patch("lambda_function.clear_pending_chunks")
    @patch("lambda_function.get_pending_chunks", return_value=["Last part."])
    def test_yes_no_continue_prompt_on_last_chunk(self, *_):
        event = make_event("IntentRequest", intent_name="AMAZON.YesIntent")
        r = lambda_function.lambda_handler(event, None)
        self.assertNotIn("continue", get_speech(r).lower())

    @patch("lambda_function.get_pending_chunks", return_value=[])
    def test_yes_with_no_pending_chunks_returns_graceful_message(self, *_):
        event = make_event("IntentRequest", intent_name="AMAZON.YesIntent")
        r = lambda_function.lambda_handler(event, None)
        self.assertIn("nothing more", get_speech(r).lower())

    @patch("lambda_function.clear_pending_chunks")
    def test_no_clears_pending_chunks(self, mock_clear):
        event = make_event("IntentRequest", intent_name="AMAZON.NoIntent", user_id="u1")
        lambda_function.lambda_handler(event, None)
        mock_clear.assert_called_once_with("u1")

    @patch("lambda_function.clear_pending_chunks")
    def test_no_keeps_session_open(self, *_):
        event = make_event("IntentRequest", intent_name="AMAZON.NoIntent")
        r = lambda_function.lambda_handler(event, None)
        self.assertFalse(r["response"]["shouldEndSession"])


# ══════════════════════════════════════════════════════════════════════════════
# 5. SetContextIntent
# ══════════════════════════════════════════════════════════════════════════════

class TestSetContextIntent(unittest.TestCase):

    @patch("lambda_function.merge_user_facts")
    @patch("lambda_function.get_user_facts", return_value={})
    def test_merges_extracted_fact(self, mock_get, mock_merge):
        event = make_event("IntentRequest", intent_name="SetContextIntent", context_value="I'm a nurse")
        r = lambda_function.lambda_handler(event, None)
        mock_merge.assert_called_once()
        self.assertIn("Got it", get_speech(r))

    @patch("lambda_function.merge_user_facts")
    def test_empty_context_slot_returns_prompt(self, mock_merge):
        event = make_event("IntentRequest", intent_name="SetContextIntent")
        r = lambda_function.lambda_handler(event, None)
        mock_merge.assert_not_called()
        self.assertIn("didn't catch", get_speech(r))

    @patch("lambda_function.merge_user_facts")
    @patch("lambda_function.get_user_facts", return_value={})
    def test_saves_to_correct_user(self, mock_get, mock_merge):
        event = make_event("IntentRequest", intent_name="SetContextIntent", context_value="I'm a pilot", user_id="user-abc")
        lambda_function.lambda_handler(event, None)
        self.assertEqual(mock_merge.call_args[0][0], "user-abc")

    @patch("lambda_function.merge_user_facts")
    @patch("lambda_function.get_user_facts", return_value={})
    def test_session_stays_open(self, *_):
        event = make_event("IntentRequest", intent_name="SetContextIntent", context_value="I like jazz")
        r = lambda_function.lambda_handler(event, None)
        self.assertFalse(r["response"]["shouldEndSession"])

    @patch("lambda_function.merge_user_facts")
    @patch("lambda_function.get_user_facts", return_value={})
    def test_extracts_job_fact(self, mock_get, mock_merge):
        event = make_event("IntentRequest", intent_name="SetContextIntent", context_value="I'm a software engineer")
        lambda_function.lambda_handler(event, None)
        fact = mock_merge.call_args[0][1]
        self.assertEqual(fact.get("job"), "software engineer")

    @patch("lambda_function.merge_user_facts")
    @patch("lambda_function.get_user_facts", return_value={})
    def test_extracts_name_fact(self, mock_get, mock_merge):
        event = make_event("IntentRequest", intent_name="SetContextIntent", context_value="my name is Nandana")
        lambda_function.lambda_handler(event, None)
        fact = mock_merge.call_args[0][1]
        self.assertEqual(fact.get("name"), "Nandana")

    @patch("lambda_function.merge_user_facts")
    @patch("lambda_function.get_user_facts", return_value={})
    def test_extracts_location_fact(self, mock_get, mock_merge):
        event = make_event("IntentRequest", intent_name="SetContextIntent", context_value="I live in New York")
        lambda_function.lambda_handler(event, None)
        fact = mock_merge.call_args[0][1]
        self.assertEqual(fact.get("location"), "New York")

    @patch("lambda_function.merge_user_facts")
    @patch("lambda_function.get_user_facts", return_value={})
    def test_unrecognised_utterance_stored_as_note(self, mock_get, mock_merge):
        event = make_event("IntentRequest", intent_name="SetContextIntent", context_value="something random")
        lambda_function.lambda_handler(event, None)
        fact = mock_merge.call_args[0][1]
        self.assertIn("note", fact)


# ══════════════════════════════════════════════════════════════════════════════
# 6. ClearContextIntent
# ══════════════════════════════════════════════════════════════════════════════

class TestClearContextIntent(unittest.TestCase):

    @patch("lambda_function.clear_user_facts")
    def test_clears_all_facts(self, mock_clear):
        event = make_event("IntentRequest", intent_name="ClearContextIntent")
        r = lambda_function.lambda_handler(event, None)
        mock_clear.assert_called_once_with("test-user-id")
        self.assertIn("cleared", get_speech(r))

    @patch("lambda_function.clear_user_facts")
    def test_clears_correct_user(self, mock_clear):
        event = make_event("IntentRequest", intent_name="ClearContextIntent", user_id="user-xyz")
        lambda_function.lambda_handler(event, None)
        mock_clear.assert_called_once_with("user-xyz")

    @patch("lambda_function.clear_user_facts")
    def test_session_stays_open(self, *_):
        event = make_event("IntentRequest", intent_name="ClearContextIntent")
        r = lambda_function.lambda_handler(event, None)
        self.assertFalse(r["response"]["shouldEndSession"])


# ══════════════════════════════════════════════════════════════════════════════
# 6b. RecallContextIntent
# ══════════════════════════════════════════════════════════════════════════════

class TestRecallContextIntent(unittest.TestCase):

    @patch("lambda_function.get_user_facts", return_value={"name": "Nandana", "job": "engineer"})
    def test_recalls_stored_facts(self, *_):
        event = make_event("IntentRequest", intent_name="RecallContextIntent")
        r = lambda_function.lambda_handler(event, None)
        speech = get_speech(r)
        self.assertIn("name", speech)
        self.assertIn("Nandana", speech)
        self.assertIn("job", speech)
        self.assertIn("engineer", speech)

    @patch("lambda_function.get_user_facts", return_value={})
    def test_no_facts_returns_graceful_message(self, *_):
        event = make_event("IntentRequest", intent_name="RecallContextIntent")
        r = lambda_function.lambda_handler(event, None)
        self.assertIn("don't have any information", get_speech(r))

    @patch("lambda_function.get_user_facts", return_value={"location": "New York"})
    def test_correct_user_id_used(self, mock_get):
        event = make_event("IntentRequest", intent_name="RecallContextIntent", user_id="u-recall")
        lambda_function.lambda_handler(event, None)
        mock_get.assert_called_once_with("u-recall")

    @patch("lambda_function.get_user_facts", return_value={"name": "Nandana"})
    def test_session_stays_open(self, *_):
        event = make_event("IntentRequest", intent_name="RecallContextIntent")
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

    def test_save_history_writes_ttl(self):
        import time
        before = int(time.time())
        self.dynamo.save_history("u1", [])
        call_kwargs = self.mock_table.update_item.call_args[1]
        ttl_value = call_kwargs["ExpressionAttributeValues"][":t"]
        # TTL should be roughly 7 days from now
        self.assertGreater(ttl_value, before + 6 * 24 * 3600)
        self.assertLess(ttl_value, before + 8 * 24 * 3600)

    def test_save_history_prunes_before_writing(self):
        import dynamo as dyn
        # Build a history that exceeds MAX_HISTORY_TURNS
        long_history = []
        for i in range(dyn.MAX_HISTORY_TURNS + 5):
            long_history.append({"role": "user", "content": f"q{i}"})
            long_history.append({"role": "assistant", "content": f"a{i}"})
        self.dynamo.save_history("u1", long_history)
        call_kwargs = self.mock_table.update_item.call_args[1]
        saved = call_kwargs["ExpressionAttributeValues"][":h"]
        self.assertEqual(len(saved), dyn.MAX_HISTORY_TURNS * 2)

    # ── prune_history ──────────────────────────────────────────────

    def test_prune_history_returns_unchanged_when_within_limit(self):
        history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        result = self.dynamo.prune_history(history, max_turns=10)
        self.assertEqual(result, history)

    def test_prune_history_trims_to_max_turns(self):
        history = []
        for i in range(25):
            history.append({"role": "user", "content": f"q{i}"})
            history.append({"role": "assistant", "content": f"a{i}"})
        result = self.dynamo.prune_history(history, max_turns=20)
        self.assertEqual(len(result), 40)  # 20 turns * 2 messages

    def test_prune_history_keeps_most_recent_turns(self):
        history = []
        for i in range(25):
            history.append({"role": "user", "content": f"q{i}"})
            history.append({"role": "assistant", "content": f"a{i}"})
        result = self.dynamo.prune_history(history, max_turns=5)
        self.assertEqual(result[0]["content"], "q20")
        self.assertEqual(result[-1]["content"], "a24")

    def test_prune_history_empty_list(self):
        self.assertEqual(self.dynamo.prune_history([], max_turns=10), [])

    def test_prune_history_exactly_at_limit(self):
        history = []
        for i in range(20):
            history.append({"role": "user", "content": f"q{i}"})
            history.append({"role": "assistant", "content": f"a{i}"})
        result = self.dynamo.prune_history(history, max_turns=20)
        self.assertEqual(len(result), 40)

    # ── get_user_facts ─────────────────────────────────────────────

    def test_get_user_facts_returns_dict_when_exists(self):
        self.mock_table.get_item.return_value = {"Item": {"userId": "u1", "userFacts": {"job": "chef"}}}
        result = self.dynamo.get_user_facts("u1")
        self.assertEqual(result, {"job": "chef"})

    def test_get_user_facts_returns_empty_dict_when_no_item(self):
        self.mock_table.get_item.return_value = {}
        result = self.dynamo.get_user_facts("u1")
        self.assertEqual(result, {})

    def test_get_user_facts_returns_empty_dict_when_no_key(self):
        self.mock_table.get_item.return_value = {"Item": {"userId": "u1"}}
        result = self.dynamo.get_user_facts("u1")
        self.assertEqual(result, {})

    # ── merge_user_facts ───────────────────────────────────────────

    def test_merge_user_facts_merges_with_existing(self):
        self.mock_table.get_item.return_value = {"Item": {"userId": "u1", "userFacts": {"name": "Alice"}}}
        self.dynamo.merge_user_facts("u1", {"job": "pilot"})
        call_kwargs = self.mock_table.update_item.call_args[1]
        saved = call_kwargs["ExpressionAttributeValues"][":f"]
        self.assertEqual(saved, {"name": "Alice", "job": "pilot"})

    def test_merge_user_facts_uses_correct_user_key(self):
        self.mock_table.get_item.return_value = {}
        self.dynamo.merge_user_facts("u-xyz", {"name": "Bob"})
        call_kwargs = self.mock_table.update_item.call_args[1]
        self.assertEqual(call_kwargs["Key"], {"userId": "u-xyz"})

    def test_merge_user_facts_overwrites_same_key(self):
        self.mock_table.get_item.return_value = {"Item": {"userId": "u1", "userFacts": {"job": "chef"}}}
        self.dynamo.merge_user_facts("u1", {"job": "pilot"})
        call_kwargs = self.mock_table.update_item.call_args[1]
        saved = call_kwargs["ExpressionAttributeValues"][":f"]
        self.assertEqual(saved["job"], "pilot")

    # ── clear_user_facts ───────────────────────────────────────────

    def test_clear_user_facts_calls_update_item(self):
        self.dynamo.clear_user_facts("u1")
        self.mock_table.update_item.assert_called_once()

    def test_clear_user_facts_uses_remove_expression(self):
        self.dynamo.clear_user_facts("u1")
        call_kwargs = self.mock_table.update_item.call_args[1]
        self.assertIn("REMOVE", call_kwargs["UpdateExpression"])

    def test_clear_user_facts_uses_correct_key(self):
        self.dynamo.clear_user_facts("user-abc")
        call_kwargs = self.mock_table.update_item.call_args[1]
        self.assertEqual(call_kwargs["Key"], {"userId": "user-abc"})

    # ── clear_user_fact (single key) ──────────────────────────────

    def test_clear_user_fact_removes_specific_key(self):
        self.mock_table.get_item.return_value = {"Item": {"userId": "u1", "userFacts": {"name": "Alice", "job": "chef"}}}
        self.dynamo.clear_user_fact("u1", "job")
        call_kwargs = self.mock_table.update_item.call_args[1]
        saved = call_kwargs["ExpressionAttributeValues"][":f"]
        self.assertNotIn("job", saved)
        self.assertIn("name", saved)

    def test_clear_user_fact_noop_when_key_absent(self):
        self.mock_table.get_item.return_value = {"Item": {"userId": "u1", "userFacts": {"name": "Alice"}}}
        self.dynamo.clear_user_fact("u1", "job")
        self.mock_table.update_item.assert_not_called()

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
# 16. VoiceProcessor
# ══════════════════════════════════════════════════════════════════════════════

from voice_processor import to_voice, chunk_text


class TestVoiceProcessor(unittest.TestCase):

    # ── pass-through ───────────────────────────────────────────────

    def test_plain_text_unchanged(self):
        self.assertEqual(to_voice("Hello there."), "Hello there.")

    def test_empty_string(self):
        self.assertEqual(to_voice(""), "")

    def test_none_returns_none(self):
        self.assertIsNone(to_voice(None))

    # ── bold / italic ──────────────────────────────────────────────

    def test_strips_double_asterisk_bold(self):
        self.assertEqual(to_voice("**important**"), "important")

    def test_strips_double_underscore_bold(self):
        self.assertEqual(to_voice("__important__"), "important")

    def test_strips_single_asterisk_italic(self):
        self.assertEqual(to_voice("*emphasis*"), "emphasis")

    def test_strips_single_underscore_italic(self):
        self.assertEqual(to_voice("_emphasis_"), "emphasis")

    def test_strips_triple_asterisk_bold_italic(self):
        self.assertEqual(to_voice("***very important***"), "very important")

    def test_bold_mid_sentence(self):
        self.assertEqual(to_voice("This is **very** important."), "This is very important.")

    # ── headers ───────────────────────────────────────────────────

    def test_strips_h1(self):
        self.assertEqual(to_voice("# Title"), "Title")

    def test_strips_h2(self):
        self.assertEqual(to_voice("## Section"), "Section")

    def test_strips_h3(self):
        self.assertEqual(to_voice("### Subsection"), "Subsection")

    # ── code ──────────────────────────────────────────────────────

    def test_strips_inline_code(self):
        self.assertEqual(to_voice("Use `print()` to output."), "Use print() to output.")

    def test_strips_fenced_code_block(self):
        text = "Here is code:\n```\nx = 1\n```\nDone."
        result = to_voice(text)
        self.assertNotIn("```", result)
        self.assertNotIn("x = 1", result)

    def test_strips_fenced_code_block_with_language(self):
        text = "Example:\n```python\nprint('hi')\n```\nEnd."
        result = to_voice(text)
        self.assertNotIn("python", result)
        self.assertNotIn("print", result)

    # ── links ─────────────────────────────────────────────────────

    def test_converts_link_to_label(self):
        self.assertEqual(to_voice("[Google](https://google.com)"), "Google")

    def test_link_mid_sentence(self):
        self.assertEqual(to_voice("Visit [our site](https://example.com) for more."), "Visit our site for more.")

    # ── numbered lists ────────────────────────────────────────────

    def test_numbered_list_two_items(self):
        text = "1. Wake up\n2. Eat breakfast"
        result = to_voice(text)
        self.assertIn("First", result)
        self.assertIn("Second", result)
        self.assertNotIn("1.", result)
        self.assertNotIn("2.", result)

    def test_numbered_list_three_items(self):
        text = "1. Step one\n2. Step two\n3. Step three"
        result = to_voice(text)
        self.assertIn("First", result)
        self.assertIn("Second", result)
        self.assertIn("Third", result)

    def test_numbered_list_items_joined_naturally(self):
        text = "1. Do this\n2. Do that"
        result = to_voice(text)
        self.assertIn("First, Do this", result)
        self.assertIn("Second, Do that", result)

    # ── bullet lists ──────────────────────────────────────────────

    def test_dash_bullet_list(self):
        text = "- Apples\n- Bananas\n- Oranges"
        result = to_voice(text)
        self.assertIn("Apples", result)
        self.assertIn("Bananas", result)
        self.assertIn("Oranges", result)
        self.assertNotIn("- ", result)

    def test_asterisk_bullet_list(self):
        text = "* First item\n* Second item"
        result = to_voice(text)
        self.assertNotIn("* ", result)
        self.assertIn("First item", result)

    def test_bullet_items_separated_by_period(self):
        text = "- One\n- Two\n- Three"
        result = to_voice(text)
        self.assertIn("One. Two. Three", result)

    # ── abbreviations ─────────────────────────────────────────────

    def test_eg_expanded(self):
        result = to_voice("Use tools, e.g. a hammer.")
        self.assertIn("for example", result)
        self.assertNotIn("e.g.", result)

    def test_ie_expanded(self):
        result = to_voice("The answer, i.e. 42.")
        self.assertIn("that is", result)

    def test_etc_expanded(self):
        result = to_voice("Bring food, water, etc.")
        self.assertIn("and so on", result)

    def test_vs_expanded(self):
        result = to_voice("Cats vs. dogs.")
        self.assertIn("versus", result)

    def test_dr_expanded(self):
        result = to_voice("Dr. Smith is here.")
        self.assertIn("Doctor Smith", result)

    # ── horizontal rules ──────────────────────────────────────────

    def test_strips_horizontal_rule_dashes(self):
        self.assertNotIn("---", to_voice("Above\n---\nBelow"))

    def test_strips_horizontal_rule_asterisks(self):
        self.assertNotIn("***", to_voice("Above\n***\nBelow"))

    # ── blockquotes ───────────────────────────────────────────────

    def test_strips_blockquote_marker(self):
        self.assertEqual(to_voice("> This is a quote."), "This is a quote.")

    # ── whitespace ────────────────────────────────────────────────

    def test_multiple_newlines_collapsed(self):
        self.assertNotIn("\n", to_voice("Line one\n\n\nLine two"))

    def test_leading_trailing_whitespace_stripped(self):
        self.assertEqual(to_voice("  hello  "), "hello")

    def test_multiple_spaces_collapsed(self):
        self.assertEqual(to_voice("too   many   spaces"), "too many spaces")

    # ── integration: realistic LLM output ─────────────────────────

    def test_realistic_markdown_response(self):
        text = (
            "## How to stay productive\n\n"
            "Here are some tips:\n"
            "1. **Wake up early** every day\n"
            "2. Use the *Pomodoro* technique, e.g. 25-minute blocks\n"
            "3. Avoid distractions, etc.\n\n"
            "Learn more at [this article](https://example.com)."
        )
        result = to_voice(text)
        self.assertNotIn("##", result)
        self.assertNotIn("**", result)
        self.assertNotIn("*", result)
        self.assertNotIn("e.g.", result)
        self.assertNotIn("etc.", result)
        self.assertNotIn("https://", result)
        self.assertIn("for example", result)
        self.assertIn("and so on", result)
        self.assertIn("this article", result)

    # ── processor applied in lambda ───────────────────────────────

    @patch("lambda_function.clear_pending_chunks")
    @patch("lambda_function.get_user_facts", return_value={})
    @patch("lambda_function.save_history")
    @patch("lambda_function.get_history", return_value=[])
    @patch("lambda_function.call_llm", return_value="**Key point:** Use `print()` to debug.")
    def test_processor_applied_to_ask_intent(self, mock_llm, *_):
        event = make_event("IntentRequest", intent_name="AskClaudeIntent", query="how do I debug")
        r = lambda_function.lambda_handler(event, None)
        speech = r["response"]["outputSpeech"]["text"]
        self.assertNotIn("**", speech)
        self.assertNotIn("`", speech)
        self.assertIn("Key point", speech)
        self.assertIn("print()", speech)

    # ── remaining abbreviations ───────────────────────────────────────────

    def test_mr_expanded(self):
        result = to_voice("Mr. Jones called.")
        self.assertIn("Mister Jones", result)
        self.assertNotIn("Mr.", result)

    def test_mrs_expanded(self):
        result = to_voice("Mrs. Smith arrived.")
        self.assertIn("Missus Smith", result)
        self.assertNotIn("Mrs.", result)

    def test_ms_expanded(self):
        result = to_voice("Ms. Lee is here.")
        self.assertIn("Miss Lee", result)
        self.assertNotIn("Ms.", result)

    def test_prof_expanded(self):
        result = to_voice("Prof. Adams teaches here.")
        self.assertIn("Professor Adams", result)
        self.assertNotIn("Prof.", result)

    def test_approx_expanded(self):
        result = to_voice("It takes approx. 10 minutes.")
        self.assertIn("approximately", result)
        self.assertNotIn("approx.", result)

    def test_est_expanded(self):
        result = to_voice("The est. cost is fifty dollars.")
        self.assertIn("estimated", result)
        self.assertNotIn("est.", result)

    def test_min_expanded(self):
        result = to_voice("Wait 5 min. please.")
        self.assertIn("minutes", result)
        self.assertNotIn("min.", result)

    def test_max_expanded(self):
        result = to_voice("The max. value is 100.")
        self.assertIn("maximum", result)
        self.assertNotIn("max.", result)

    def test_no_expanded(self):
        result = to_voice("See item no. 5.")
        self.assertIn("number", result)
        self.assertNotIn("no.", result)

    # ── bullet symbol (•) ────────────────────────────────────────────────

    def test_bullet_symbol_list(self):
        text = "• Apples\n• Bananas\n• Oranges"
        result = to_voice(text)
        self.assertNotIn("•", result)
        self.assertIn("Apples", result)
        self.assertIn("Bananas", result)

    # ── numbered list edge cases ─────────────────────────────────────────

    def test_numbered_list_beyond_ten_uses_item_n(self):
        lines = "\n".join(f"{i}. Item {i}" for i in range(1, 12))
        result = to_voice(lines)
        # Item 11 should be labelled with the "Item N" fallback prefix
        self.assertIn("Item 11, Item 11", result)

    def test_numbered_list_followed_by_paragraph(self):
        text = "Steps:\n1. First step\n2. Second step\n\nDone."
        result = to_voice(text)
        self.assertIn("First, First step", result)
        self.assertIn("Second, Second step", result)
        self.assertIn("Done", result)


# ══════════════════════════════════════════════════════════════════════════════
# 17. ChunkText
# ══════════════════════════════════════════════════════════════════════════════

class TestChunkText(unittest.TestCase):

    def test_short_text_returns_single_chunk(self):
        text = "Hello there."
        self.assertEqual(chunk_text(text), [text])

    def test_empty_string_returns_single_chunk(self):
        self.assertEqual(chunk_text(""), [""])

    def test_text_exactly_at_limit_is_single_chunk(self):
        text = "a" * 750
        self.assertEqual(chunk_text(text, limit=750), [text])

    def test_long_text_splits_into_multiple_chunks(self):
        text = "a" * 800
        chunks = chunk_text(text, limit=750)
        self.assertGreater(len(chunks), 1)

    def test_each_chunk_within_limit(self):
        text = ("word " * 200).strip()
        for chunk in chunk_text(text, limit=100):
            self.assertLessEqual(len(chunk), 100)

    def test_splits_at_sentence_boundary(self):
        sentence = "This is a sentence. "
        text = sentence * 40  # well over 750 chars
        chunks = chunk_text(text)
        for chunk in chunks:
            # each chunk should end cleanly, not mid-word
            self.assertFalse(chunk.endswith(" "))

    def test_no_content_lost(self):
        words = ["word"] * 300
        text = " ".join(words)
        chunks = chunk_text(text, limit=100)
        rejoined = " ".join(chunks)
        # every word should still be present
        self.assertEqual(rejoined.count("word"), 300)

    def test_custom_limit_respected(self):
        text = "Hello world. " * 10
        chunks = chunk_text(text, limit=50)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 50)

    def test_single_very_long_word_hard_cuts(self):
        # No whitespace or sentence boundary — must hard-cut at limit
        text = "a" * 1600
        chunks = chunk_text(text, limit=750)
        self.assertEqual(len(chunks), 3)
        for chunk in chunks[:-1]:
            self.assertEqual(len(chunk), 750)


# ══════════════════════════════════════════════════════════════════════════════
# 18. Logger
# ══════════════════════════════════════════════════════════════════════════════

import io
from logger import log_invocation, emit_metrics


class TestLogger(unittest.TestCase):

    def _captured_log(self, fn, *args, **kwargs):
        """Call fn and return all lines printed to stdout."""
        buf = io.StringIO()
        with patch("builtins.print", side_effect=lambda s: buf.write(s + "\n")):
            fn(*args, **kwargs)
        return [json.loads(line) for line in buf.getvalue().splitlines() if line]

    # ── log_invocation ────────────────────────────────────────────────────

    def test_log_invocation_emits_json(self):
        lines = self._captured_log(log_invocation, "AskClaudeIntent", "u1", 120)
        self.assertEqual(len(lines), 1)

    def test_log_invocation_contains_required_fields(self):
        lines = self._captured_log(log_invocation, "AskClaudeIntent", "u1", 120)
        entry = lines[0]
        self.assertEqual(entry["event"], "invocation")
        self.assertEqual(entry["intent"], "AskClaudeIntent")
        self.assertEqual(entry["userId"], "u1")
        self.assertEqual(entry["latency_ms"], 120)
        self.assertIsNone(entry["error"])

    def test_log_invocation_includes_error_string(self):
        lines = self._captured_log(log_invocation, "AskClaudeIntent", "u1", 50, error=ValueError("boom"))
        self.assertIn("boom", lines[0]["error"])

    def test_log_invocation_null_error_when_no_error(self):
        lines = self._captured_log(log_invocation, "AskClaudeIntent", "u1", 50)
        self.assertIsNone(lines[0]["error"])

    # ── emit_metrics ──────────────────────────────────────────────────────

    def test_emit_metrics_emits_valid_emf(self):
        lines = self._captured_log(emit_metrics, "AskClaudeIntent", 200)
        entry = lines[0]
        self.assertIn("_aws", entry)
        self.assertIn("CloudWatchMetrics", entry["_aws"])

    def test_emit_metrics_correct_namespace(self):
        lines = self._captured_log(emit_metrics, "AskClaudeIntent", 200)
        ns = lines[0]["_aws"]["CloudWatchMetrics"][0]["Namespace"]
        self.assertEqual(ns, "AlexaLLM")

    def test_emit_metrics_intent_dimension(self):
        lines = self._captured_log(emit_metrics, "SetContextIntent", 80)
        self.assertEqual(lines[0]["Intent"], "SetContextIntent")

    def test_emit_metrics_error_flag_set_on_failure(self):
        lines = self._captured_log(emit_metrics, "AskClaudeIntent", 50, error=Exception("fail"))
        self.assertEqual(lines[0]["Errors"], 1)

    def test_emit_metrics_error_flag_zero_on_success(self):
        lines = self._captured_log(emit_metrics, "AskClaudeIntent", 50)
        self.assertEqual(lines[0]["Errors"], 0)

    def test_emit_metrics_latency_recorded(self):
        lines = self._captured_log(emit_metrics, "AskClaudeIntent", 350)
        self.assertEqual(lines[0]["Latency"], 350)

    # ── handler emits logs ────────────────────────────────────────────────

    @patch("lambda_function.emit_metrics")
    @patch("lambda_function.log_invocation")
    @patch("lambda_function.clear_pending_chunks")
    @patch("lambda_function.get_user_facts", return_value={})
    @patch("lambda_function.save_history")
    @patch("lambda_function.get_history", return_value=[])
    @patch("lambda_function.call_llm", return_value="Answer.")
    def test_handler_calls_log_invocation(self, mock_llm, mock_gh, mock_sh, mock_guf, mock_cpc, mock_log, mock_emit):
        event = make_event("IntentRequest", intent_name="AskClaudeIntent", query="test")
        lambda_function.lambda_handler(event, None)
        mock_log.assert_called_once()
        args = mock_log.call_args[0]
        self.assertEqual(args[0], "AskClaudeIntent")

    @patch("lambda_function.emit_metrics")
    @patch("lambda_function.log_invocation")
    @patch("lambda_function.clear_pending_chunks")
    @patch("lambda_function.get_user_facts", return_value={})
    @patch("lambda_function.save_history")
    @patch("lambda_function.get_history", return_value=[])
    @patch("lambda_function.call_llm", return_value="Answer.")
    def test_handler_calls_emit_metrics(self, mock_llm, mock_gh, mock_sh, mock_guf, mock_cpc, mock_log, mock_emit):
        event = make_event("IntentRequest", intent_name="AskClaudeIntent", query="test")
        lambda_function.lambda_handler(event, None)
        mock_emit.assert_called_once()

    @patch("lambda_function.emit_metrics")
    @patch("lambda_function.log_invocation")
    def test_handler_logs_launch_request(self, mock_log, mock_emit):
        event = make_event("LaunchRequest")
        lambda_function.lambda_handler(event, None)
        mock_log.assert_called_once()
        self.assertEqual(mock_log.call_args[0][0], "LaunchRequest")


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
