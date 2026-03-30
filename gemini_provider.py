import json
import os
import urllib.request
import urllib.error


def ask_llm(user_message, conversation_history=None):
    if conversation_history is None:
        conversation_history = []

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    contents = []

    for msg in conversation_history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({
            "role": role,
            "parts": [{"text": msg["content"]}],
        })

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
        f"gemini-2.0-flash:generateContent?key={api_key}"
    )

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
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
