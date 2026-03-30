import json
import os
import urllib.request
import urllib.error


def ask_llm(user_message, conversation_history=None):
    if conversation_history is None:
        conversation_history = []

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    system_message = {
        "role": "system",
        "content": (
            "You are a helpful voice assistant speaking through an Alexa device. "
            "Keep your responses concise and conversational — ideally under 3 sentences unless the user asks for detail. "
            "Don't use markdown, bullet points, or formatting — this will be spoken aloud. "
            "Be warm, direct, and natural."
        ),
    }

    messages = [system_message] + conversation_history + [{"role": "user", "content": user_message}]

    payload = {
        "model": "mistralai/mistral-7b-instruct:free",
        "max_tokens": 300,
        "temperature": 0.7,
        "messages": messages,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://alexa-llm.app",
        "X-Title": "Alexa Smart Assistant",
    }

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=25) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        print(f"OpenRouter API error: {e.code} {error_body}")
        raise
