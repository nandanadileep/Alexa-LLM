"""
Alexa Smart Assistant - Lambda Function

Set the LLM_PROVIDER environment variable to switch backends:
    LLM_PROVIDER=groq    → Groq (Llama 3.3 70B) — requires GROQ_API_KEY
    LLM_PROVIDER=gemini  → Google Gemini 2.5 Flash — requires GEMINI_API_KEY
"""

import json
import os

from dynamo import get_history, save_history

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq").lower()

if LLM_PROVIDER == "gemini":
    from gemini_provider import ask_llm
elif LLM_PROVIDER == "openrouter":
    from openrouter_provider import ask_llm
else:
    from groq_provider import ask_llm


# ─── Alexa Response Builders ────────────────────────────────────────

def build_response(speech_text, session_attributes=None, should_end=False):
    if session_attributes is None:
        session_attributes = {}

    response = {
        "version": "1.0",
        "sessionAttributes": session_attributes,
        "response": {
            "outputSpeech": {
                "type": "PlainText",
                "text": speech_text,
            },
            "shouldEndSession": should_end,
        },
    }

    if not should_end:
        response["response"]["reprompt"] = {
            "outputSpeech": {
                "type": "PlainText",
                "text": "Is there anything else you'd like to ask?",
            }
        }

    return response


# ─── Intent Handlers ────────────────────────────────────────────────

def handle_ask_intent(event):
    slots = event["request"].get("intent", {}).get("slots", {})
    user_query = (
        slots.get("query", {}).get("value")
        or slots.get("question", {}).get("value")
    )

    if not user_query:
        return build_response("I didn't catch that. Could you ask your question again?")

    user_id = event["session"]["user"]["userId"]
    conversation_history = get_history(user_id)

    try:
        llm_response = ask_llm(user_query, conversation_history)

        updated_history = conversation_history + [
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": llm_response},
        ]
        save_history(user_id, updated_history)

        return build_response(llm_response)

    except Exception as e:
        print(f"Error calling LLM: {e}")
        return build_response(
            "Sorry, I had trouble getting a response. Please try again in a moment."
        )


def handle_yes_no_intent(event, word):
    user_id = event["session"]["user"]["userId"]
    conversation_history = get_history(user_id)

    try:
        llm_response = ask_llm(word, conversation_history)
        updated_history = conversation_history + [
            {"role": "user", "content": word},
            {"role": "assistant", "content": llm_response},
        ]
        save_history(user_id, updated_history)
        return build_response(llm_response)
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return build_response("Sorry, I had trouble getting a response. Please try again in a moment.")


def handle_launch_request():
    return build_response("Hey! I'm your AI-powered assistant. Ask me anything.")


def handle_help_intent():
    return build_response(
        "Just ask me any question and I'll give you a thoughtful answer. "
        "For example, try asking me to explain something, help you think through a problem, "
        "or answer a question."
    )


def handle_stop_intent():
    return build_response("Goodbye!", should_end=True)


def handle_fallback():
    return build_response(
        "I didn't understand that. Try asking me a question — like "
        "'What's the best way to learn Python?' or 'Explain how the internet works.'"
    )


# ─── Main Lambda Handler ────────────────────────────────────────────

def lambda_handler(event, context):
    print(f"Received event: {json.dumps(event, indent=2)}")

    request_type = event["request"]["type"]

    if request_type == "LaunchRequest":
        return handle_launch_request()

    elif request_type == "IntentRequest":
        intent_name = event["request"]["intent"]["name"]

        if intent_name == "AskClaudeIntent":
            return handle_ask_intent(event)
        elif intent_name == "AMAZON.YesIntent":
            return handle_yes_no_intent(event, "yes")
        elif intent_name == "AMAZON.NoIntent":
            return handle_yes_no_intent(event, "no")
        elif intent_name == "AMAZON.HelpIntent":
            return handle_help_intent()
        elif intent_name in ("AMAZON.StopIntent", "AMAZON.CancelIntent"):
            return handle_stop_intent()
        elif intent_name == "AMAZON.FallbackIntent":
            return handle_fallback()
        else:
            return handle_fallback()

    elif request_type == "SessionEndedRequest":
        print(f"Session ended: {event['request'].get('reason')}")
        return build_response("", should_end=True)

    else:
        return handle_fallback()
