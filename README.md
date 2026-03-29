# Alexa Smart Assistant

> Make your Alexa actually intelligent by connecting it to powerful LLMs.

## What is this?

A custom Alexa skill that routes your voice queries to an LLM (currently Gemini / Groq / Claude) instead of Alexa's default responses. Think of it as replacing Alexa's brain while keeping its ears and mouth.

**Architecture:**
```
Your Voice → Alexa (ears) → AWS Lambda (router) → LLM API (brain) → Alexa (mouth) → You hear the answer
```

## Features

- 🎙️ Voice-powered AI assistant through any Alexa device
- 🧠 Powered by your choice of LLM (Gemini, Groq, Claude)
- 💬 Multi-turn conversations — remembers context within a session
- 🆓 Runs entirely on free tiers (Lambda free tier + free LLM API)
- 🔧 You own the stack — customize system prompts, swap models, add memory

## Project Structure

```
├── lambda/
│   ├── lambda_function.py      # Main Lambda handler (Groq/Llama)
│   └── lambda_function_gemini.py # Alternative: Google Gemini version
├── skill-config/
│   └── interaction-model.json  # Alexa skill interaction model
├── SETUP-GUIDE.md              # Step-by-step deployment guide
└── README.md
```

## Quick Start

1. Get an API key ([Groq](https://console.groq.com) or [Google AI Studio](https://aistudio.google.com))
2. Create an AWS Lambda function (Python 3.13)
3. Paste the Lambda code and set your API key as environment variable
4. Create an Alexa skill and paste the interaction model
5. Connect the skill to your Lambda

Full instructions in [SETUP-GUIDE.md](./SETUP-GUIDE.md)

## Cost

| Service | Cost |
|---------|------|
| AWS Lambda | Free (1M requests/month, forever) |
| Alexa Skill | Free |
| Groq API | Free tier (no credit card) |
| Gemini API | Free tier (no credit card) |

**Total: $0/month** for personal use.

## Roadmap

- [x] Phase 1: Basic Alexa → LLM pipeline
- [ ] Phase 2: Persistent memory across sessions (DynamoDB)
- [ ] Phase 3: Context-aware personal assistant (custom memory system)
- [ ] Phase 4: Multi-user support and productization

## Supported LLM Providers

| Provider | Model | Status |
|----------|-------|--------|
| Groq | Llama 3.3 70B | ✅ Working |
| Google | Gemini 2.5 Flash | 🔄 Ready |
| Anthropic | Claude Haiku | ✅ Working (needs credits) |

## License

MIT
