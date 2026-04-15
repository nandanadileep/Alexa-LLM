"""
Alexa Smart Assistant - Lambda Function

Set the LLM_PROVIDER environment variable to switch backends:
    LLM_PROVIDER=groq    → Groq (Llama 3.3 70B) — requires GROQ_API_KEY
    LLM_PROVIDER=gemini  → Google Gemini 2.5 Flash — requires GEMINI_API_KEY
"""

import json
import os

import re

from dynamo import (
    get_history, save_history,
    get_user_facts, merge_user_facts, clear_user_facts,
    get_pending_chunks, save_pending_chunks, clear_pending_chunks,
)
from voice_processor import to_voice, chunk_text
from llm_caller import call_llm


# ─── User Fact Helpers ──────────────────────────────────────────────

_FACT_PATTERNS = [
    (r"my name is (.+)",                              "name"),
    (r"i(?:'m| am) called (.+)",                      "name"),
    (r"i(?:'m| am) (\d+) years? old",                "age"),
    (r"my age is (\d+)",                              "age"),
    (r"i(?:'m| am) from (.+)",                        "location"),
    (r"i live in (.+)",                               "location"),
    (r"i(?:'m| am) based in (.+)",                    "location"),
    (r"i work(?:ing)? as (?:a |an )?(.+)",            "job"),
    (r"my (?:job|profession|occupation|role) is (.+)", "job"),
    (r"i(?:'m| am) a(?:n)? (.+)",                    "job"),
    (r"i prefer (.+)",                                "preference"),
    (r"i like (.+)",                                  "preference"),
]


def _extract_fact(utterance):
    text = utterance.strip()
    # Strip leading "remember [that]" prefix Alexa may pass through
    text = re.sub(r"^(?:remember\s+(?:that\s+)?)", "", text, flags=re.IGNORECASE)
    for pattern, key in _FACT_PATTERNS:
        m = re.match(pattern, text, re.IGNORECASE)
        if m:
            value = m.group(1).strip().rstrip(".,!?")
            return {key: value}
    return {"note": text.strip()}


def _format_facts(facts):
    if not facts:
        return ""
    return "; ".join(f"{k}: {v}" for k, v in facts.items())


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
    user_context = _format_facts(get_user_facts(user_id))

    try:
        llm_response = call_llm(user_query, conversation_history, user_context)

        updated_history = conversation_history + [
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": llm_response},
        ]
        save_history(user_id, updated_history)

        chunks = chunk_text(to_voice(llm_response))
        if len(chunks) == 1:
            clear_pending_chunks(user_id)
            return build_response(chunks[0])

        save_pending_chunks(user_id, chunks[1:])
        return build_response(
            chunks[0] + " ... Want me to continue?",
        )

    except Exception as e:
        print(f"Error calling LLM: {e}")
        return build_response(
            "Sorry, I had trouble getting a response. Please try again in a moment."
        )


def handle_continue_intent(event):
    user_id = event["session"]["user"]["userId"]
    chunks = get_pending_chunks(user_id)

    if not chunks:
        return build_response("There's nothing more to continue. Feel free to ask me something else.")

    current = chunks[0]
    remaining = chunks[1:]

    if remaining:
        save_pending_chunks(user_id, remaining)
        return build_response(current + " ... Want me to continue?")

    clear_pending_chunks(user_id)
    return build_response(current)


def handle_yes_no_intent(event, word):
    user_id = event["session"]["user"]["userId"]
    conversation_history = get_history(user_id)
    user_context = _format_facts(get_user_facts(user_id))

    try:
        llm_response = call_llm(word, conversation_history, user_context)
        updated_history = conversation_history + [
            {"role": "user", "content": word},
            {"role": "assistant", "content": llm_response},
        ]
        save_history(user_id, updated_history)
        should_end = word == "no"
        return build_response(to_voice(llm_response), should_end=should_end)
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return build_response("Sorry, I had trouble getting a response. Please try again in a moment.")


def handle_set_context_intent(event):
    slots = event["request"].get("intent", {}).get("slots", {})
    context_text = slots.get("context", {}).get("value")

    if not context_text:
        return build_response("I didn't catch that. Try saying something like, remember that I'm a software engineer who prefers short answers.")

    user_id = event["session"]["user"]["userId"]
    fact = _extract_fact(context_text)
    merge_user_facts(user_id, fact)
    key, value = next(iter(fact.items()))
    return build_response(f"Got it. I've noted your {key} as {value}.")


def handle_recall_context_intent(event):
    user_id = event["session"]["user"]["userId"]
    facts = get_user_facts(user_id)

    if not facts:
        return build_response("I don't have any information stored about you yet.")

    parts = [f"{k}: {v}" for k, v in facts.items()]
    return build_response("Here's what I know about you. " + ". ".join(parts) + ".")


def handle_clear_context_intent(event):
    user_id = event["session"]["user"]["userId"]
    clear_user_facts(user_id)
    return build_response("Done. I've cleared everything I know about you.")


def handle_launch_request():
    return build_response("Hey! I'm Kimi. What's on your mind?")


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
        elif intent_name == "ContinueIntent":
            return handle_continue_intent(event)
        elif intent_name == "SetContextIntent":
            return handle_set_context_intent(event)
        elif intent_name == "RecallContextIntent":
            return handle_recall_context_intent(event)
        elif intent_name == "ClearContextIntent":
            return handle_clear_context_intent(event)
        elif intent_name == "AMAZON.YesIntent":
            return handle_continue_intent(event)
        elif intent_name == "AMAZON.NoIntent":
            user_id = event["session"]["user"]["userId"]
            clear_pending_chunks(user_id)
            return build_response("Alright, let me know if you have another question.", should_end=False)
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
