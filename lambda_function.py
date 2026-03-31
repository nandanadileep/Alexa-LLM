"""
Alexa Smart Assistant - Lambda Function

Set the LLM_PROVIDER environment variable to switch backends:
    LLM_PROVIDER=groq    → Groq (Llama 3.3 70B) — requires GROQ_API_KEY
    LLM_PROVIDER=gemini  → Google Gemini 2.5 Flash — requires GEMINI_API_KEY
"""

import json
import os

from dynamo import get_history, save_history, get_user_context, save_user_context, clear_user_context

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
    user_context = get_user_context(user_id)

    try:
        llm_response = ask_llm(user_query, conversation_history, user_context)

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
    user_context = get_user_context(user_id)

    try:
        llm_response = ask_llm(word, conversation_history, user_context)
        updated_history = conversation_history + [
            {"role": "user", "content": word},
            {"role": "assistant", "content": llm_response},
        ]
        save_history(user_id, updated_history)
        return build_response(llm_response)
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return build_response("Sorry, I had trouble getting a response. Please try again in a moment.")


def handle_set_context_intent(event):
    slots = event["request"].get("intent", {}).get("slots", {})
    context_text = slots.get("context", {}).get("value")

    if not context_text:
        return build_response("I didn't catch that. Try saying something like, remember that I'm a software engineer who prefers short answers.")

    user_id = event["session"]["user"]["userId"]
    save_user_context(user_id, context_text)
    return build_response(f"Got it. I'll keep that in mind going forward.")


def handle_clear_context_intent(event):
    user_id = event["session"]["user"]["userId"]
    clear_user_context(user_id)
    return build_response("Done. I've cleared your personal context.")


def handle_launch_request():
    return build_response("Hey! I'm Kimi. Ask me anything.")


def handle_help_intent():
    return build_response(
        "Just ask me any question and I'll give you a thoughtful answer. "
        "For example, try asking me to explain something, help you think through a problem, "
        "or answer a question."
    )


def handle_stop_intent():
    return build_response("Goodbye!", should_end=True)


def handle_fallback():
    fallbacks = [
        "How are you feeling about that?",
        "Tell me more, I'm listening.",
        "How did that make you feel?",
        "What's on your mind?",
        "I'm here. What would you like to talk about?",
        "How are you feeling right now?",
        "Want to talk about it?",
    ]
    import random
    return build_response(random.choice(fallbacks))


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
        elif intent_name == "SetContextIntent":
            return handle_set_context_intent(event)
        elif intent_name == "ClearContextIntent":
            return handle_clear_context_intent(event)
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
