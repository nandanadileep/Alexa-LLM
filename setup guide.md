# Alexa-Claude Bridge — Setup Guide

## What You're Building
Alexa (voice input) → AWS Lambda → Claude API → Alexa (voice response)

---

## Step 1: Deploy the Lambda Function

### 1.1 — Create the Lambda Function
1. Go to [AWS Lambda Console](https://console.aws.amazon.com/lambda/)
2. Make sure your region is set to **US East (N. Virginia) — us-east-1** (top right corner)
3. Click **"Create function"**
4. Choose **"Author from scratch"**
5. Settings:
   - Function name: `alexa-claude-bridge`
   - Runtime: **Python 3.13**
   - Architecture: **arm64** (cheaper)
6. Click **"Create function"**

### 1.2 — Upload the Code
1. In your Lambda function page, scroll to the **"Code"** section
2. You'll see a file called `lambda_function.py` — click on it
3. Delete all the existing code
4. Copy-paste the entire content of `lambda/lambda_function.py` from this project
5. Click **"Deploy"**

### 1.3 — Set the API Key
1. Go to the **"Configuration"** tab → **"Environment variables"**
2. Click **"Edit"** → **"Add environment variable"**
3. Key: `ANTHROPIC_API_KEY`
4. Value: Your Claude API key from [console.anthropic.com](https://console.anthropic.com)
5. Click **"Save"**

### 1.4 — Increase Timeout
1. Still in **"Configuration"** tab → **"General configuration"**
2. Click **"Edit"**
3. Set timeout to **30 seconds** (Claude needs a few seconds to respond)
4. Set memory to **256 MB**
5. Click **"Save"**

### 1.5 — Add Alexa Trigger
1. Go to the **"Function overview"** section at the top
2. Click **"Add trigger"**
3. Select **"Alexa Skills Kit"**
4. For now, select **"Disable"** skill ID verification (we'll enable it later)
5. Click **"Add"**
6. **Copy your Lambda function ARN** from the top right — you'll need it in Step 2

---

## Step 2: Create the Alexa Skill

### 2.1 — Create a New Skill
1. Go to [Alexa Developer Console](https://developer.amazon.com/alexa/console/ask)
2. Click **"Create Skill"**
3. Settings:
   - Skill name: `Claude Assistant`
   - Primary locale: **English (IN)** (since you're in India) or English (US)
   - Choose model: **Custom**
   - Hosting: **Provision your own** (we're using our Lambda)
4. Click **"Create Skill"**
5. Choose **"Start from Scratch"** template

### 2.2 — Set Up the Interaction Model
1. In the left sidebar, click **"Interaction Model"** → **"JSON Editor"**
2. Paste the entire content of `skill-config/interaction-model.json`
3. Click **"Save"**
4. Click **"Build Skill"** (top of page) — wait for it to complete

### 2.3 — Connect to Lambda
1. In the left sidebar, click **"Endpoint"**
2. Select **"AWS Lambda ARN"**
3. In the **"Default Region"** field, paste your Lambda ARN from Step 1.6
4. Click **"Save Endpoints"**

### 2.4 — Get Your Skill ID (for Lambda security)
1. Go back to the skill list page
2. Click **"View Skill ID"** under your skill
3. Copy the Skill ID (starts with `amzn1.ask.skill.`)
4. Go back to your Lambda function → Triggers → Alexa Skills Kit
5. Edit the trigger, paste the Skill ID, and enable verification
6. Save

---

## Step 3: Test It!

### Option A — Alexa Developer Console (no device needed!)
1. In the Alexa Developer Console, click the **"Test"** tab
2. Change the dropdown from "Off" to **"Development"**
3. Type or speak: **"open claude assistant"**
4. Then ask: **"explain quantum computing in simple terms"**

### Option B — Your Physical Alexa Device
- Your Alexa skill is automatically available on any Alexa device 
  linked to the same Amazon account you used for the developer console
- Say: **"Alexa, open claude assistant"**
- Then ask your question!

---

## How It Works

```
You: "Alexa, open claude assistant"
Alexa: "Hey! I'm your Claude-powered assistant. Ask me anything."

You: "What's the best way to learn Rust?"
   → Alexa captures voice → converts to text
   → Sends to Lambda function
   → Lambda calls Claude API with your question
   → Claude responds with a concise answer
   → Lambda sends response back to Alexa
   → Alexa speaks the answer
Alexa: "The best way to learn Rust is to start with..."

You: "Can you elaborate on that?"
   → Session history is maintained, so Claude knows the context!
```

---

## Cost Breakdown (for your peace of mind)

| Service        | Free Tier                        | Your Likely Usage    |
|----------------|----------------------------------|----------------------|
| AWS Lambda     | 1M requests/month free           | Maybe 1,000/month    |
| Claude Haiku   | ~$0.25 per 1M input tokens       | Pennies per day      |
| Alexa Skill    | Completely free to create        | $0                   |

**Estimated monthly cost: Less than $1** (mostly Claude API calls)

---

## Troubleshooting

**"There was a problem with the requested skill's response"**
- Check Lambda logs in CloudWatch
- Make sure ANTHROPIC_API_KEY is set correctly
- Make sure timeout is 30 seconds

**Alexa says nothing / skill closes immediately**
- Check that your Lambda ARN is correct in the Alexa endpoint
- Make sure the interaction model built successfully

**Claude responses are too long / get cut off**
- The code already limits to 300 tokens
- You can lower `max_tokens` in index.mjs for shorter responses

---

## Next Steps (Phase 2 & 3)

Once this is working:
- [ ] Apply for AWS Activate credits ($1,000 free)
- [ ] Apply for Anthropic startup program ($25K Claude credits)  
- [ ] Apply for Google for Startups ($2,000 + $10K Claude credits)
- [ ] Add your context-aware memory system (JSON from your project)
- [ ] Switch to Sonnet for complex queries, keep Haiku for simple ones
- [ ] Add persistent memory across sessions (DynamoDB)
