"""
Alexa Smart Assistant - Lambda Function (Gemini Version)

Powered by Google Gemini 2.5 Flash (free tier, no credit card needed).
Get your API key at: https://aistudio.google.com

Environment Variables Required:
    GEMINI_API_KEY - Your API key from Google AI Studio
"""

import json
import os
import urllib.request
import urllib.error


# ─── Gemini API Call (Free) ──────────────────────────────────────────

def ask_llm(user_message, conversation_history=None):
    """Send a message to Gemini and get a response."""
    if conversation_history is None:
        conversation_history = []

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    # Build contents array with conversation history
    contents = []

    # Add conversation history
    for msg in conversation_history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({
            "role": role,
            "parts": [{"text": msg["content"]}],
        })

    # Add current user message
    contents.append({
        "role": "user",
        "parts": [{"text": user_message}],
    })

    payload = {
        "contents": contents,
        "systemInstruction": {
            "parts": [
                {
                    "text": (
                        "You are a helpful voice assistant speaking through an Alexa device. "
                        "Keep your responses concise and conversational — ideally under 3 sentences unless the user asks for detail. "
                        "Don't use markdown, bullet points, or formatting — this will be spoken aloud. "
                        "Be warm, direct, and natural."
                    )
                }
            ]
        },
        "generationConfig": {
            "maxOutputTokens": 300,
            "temperature": 0.7,
        },
    }

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"
    )

    headers = {
        "Content-Type": "application/json",
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=25) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data["candidates"][0]["content"]["parts"][0]["text"]
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        print(f"Gemini API error: {e.code} {error_body}")
        raise


# ─── Alexa Response Builders ────────────────────────────────────────

def build_response(speech_text, session_attributes=None, should_end=False):
    """Build an Alexa-compatible JSON response."""
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
    """Handle the main AskClaudeIntent — send user's question to the LLM."""
    slots = event["request"].get("intent", {}).get("slots", {})
    user_query = (
        slots.get("query", {}).get("value")
        or slots.get("question", {}).get("value")
    )

    if not user_query:
        return build_response("I didn't catch that. Could you ask your question again?")

    # Retrieve conversation history from session
    session_attributes = event.get("session", {}).get("attributes") or {}
    conversation_history = session_attributes.get("conversationHistory", [])

    try:
        llm_response = ask_llm(user_query, conversation_history)

        # Update conversation history (keep last 6 turns to stay within limits)
        updated_history = conversation_history + [
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": llm_response},
        ]
        updated_history = updated_history[-6:]

        return build_response(
            llm_response,
            {"conversationHistory": updated_history},
        )

    except Exception as e:
        print(f"Error calling LLM: {e}")
        return build_response(
            "Sorry, I had trouble getting a response. Please try again in a moment."
        )


def handle_launch_request():
    """Handle when the user opens the skill."""
    return build_response("Hey! I'm your AI-powered assistant. Ask me anything.")


def handle_help_intent():
    """Handle AMAZON.HelpIntent."""
    return build_response(
        "Just ask me any question and I'll give you a thoughtful answer. "
        "For example, try asking me to explain something, help you think through a problem, "
        "or answer a question."
    )


def handle_stop_intent():
    """Handle AMAZON.StopIntent and AMAZON.CancelIntent."""
    return build_response("Goodbye!", should_end=True)


def handle_fallback():
    """Handle AMAZON.FallbackIntent and unknown intents."""
    return build_response(
        "I didn't understand that. Try asking me a question — like "
        "'What's the best way to learn Python?' or 'Explain how the internet works.'"
    )


# ─── Main Lambda Handler ────────────────────────────────────────────

def lambda_handler(event, context):
    """Main entry point for the Lambda function."""
    print(f"Received event: {json.dumps(event, indent=2)}")

    request_type = event["request"]["type"]

    if request_type == "LaunchRequest":
        return handle_launch_request()

    elif request_type == "IntentRequest":
        intent_name = event["request"]["intent"]["name"]

        if intent_name == "AskClaudeIntent":
            return handle_ask_intent(event)
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
