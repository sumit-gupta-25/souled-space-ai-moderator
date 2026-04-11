from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os
import re

app = FastAPI()

# HuggingFace Token
HF_TOKEN = os.getenv("HF_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# Hinglish abuse words
BAD_WORDS = {
    "chutiya", "madarchod", "bhosdike", "gandu", "bc", "mc",
    "randi", "harami", "loda", "lund", "chodu", "bhenchod",
    "behenchod", "gand", "chut", "bhosda", "haramzada",
    "lodu", "chutiye", "madarchode"
}

# Mental harm phrases
HARMFUL_PHRASES = [
    "go die", "kill yourself", "you should die", "just die", "drop dead",
    "you deserve to die", "end your life",

    "you are worthless", "you are useless", "you are nothing",
    "you are a failure", "you mean nothing", "you are nobody",

    "nobody likes you", "no one likes you", "no one cares about you",
    "everyone hates you", "you are unwanted",

    "you should disappear", "just disappear", "leave this world",
    "no one would miss you",

    "why are you even alive", "you should not exist",
    "everything is your fault", "you ruin everything"
]

class TextInput(BaseModel):
    text: str


# Clean text
def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text


# API call
def query_model(model, text):
    try:
        url = f"https://api-inference.huggingface.co/models/{model}"
        response = requests.post(
            url,
            headers=HEADERS,
            json={"inputs": text},
            timeout=5
        )
        return response.json()
    except:
        return []


# Custom abuse detection
def has_custom_abuse(text: str) -> bool:
    words = clean_text(text).split()
    return any(word in BAD_WORDS for word in words)


# Harmful phrase detection
def has_harmful_phrase(text: str) -> bool:
    text = text.lower()
    return any(phrase in text for phrase in HARMFUL_PHRASES)


# Pattern-based detection
def detect_harmful_patterns(text: str) -> bool:
    text = text.lower()

    patterns = [
        ("you are", ["worthless", "useless", "nothing", "failure"]),
        ("no one", ["likes you", "cares about you"]),
        ("everyone", ["hates you"]),
        ("you should", ["die", "disappear", "not exist"]),
    ]

    for prefix, words in patterns:
        if prefix in text:
            for word in words:
                if word in text:
                    return True

    return False


# MAIN LOGIC
def smart_moderation(text: str):

    # 1. Custom abuse (highest priority)
    if has_custom_abuse(text):
        return {
            "decision": "block",
            "severity": "high",
            "reason": "abusive language"
        }

    # 2. Mental harm detection
    if has_harmful_phrase(text) or detect_harmful_patterns(text):
        return {
            "decision": "block",
            "severity": "high",
            "reason": "harmful content affecting mental health"
        }

    # 3. AI Models
    toxic_response = query_model("unitary/multilingual-toxic-xlm-roberta", text)
    hate_response = query_model("Hate-speech-CNERG/bert-base-uncased-hatexplain", text)

    if not toxic_response or not hate_response:
        return {"decision": "allow", "severity": "none"}

    try:
        # FIXED MULTI-LABEL PARSING
        toxic_score = 0.0

        for item in toxic_response:
            label = item.get("label", "").lower()
            score = float(item.get("score", 0))

            if label in ["toxic", "insult", "threat", "obscene"]:
                toxic_score = max(toxic_score, score)

        hate_label = hate_response[0].get("label", "").lower()
        hate_score = float(hate_response[0].get("score", 0))

    except:
        return {"decision": "allow", "severity": "none"}

    # Noise filter (allow emotional expression)
    if toxic_score < 0.2 and hate_score < 0.2:
        return {"decision": "allow", "severity": "none"}

    # Strong toxicity
    if toxic_score > 0.9:
        return {
            "decision": "block",
            "severity": "high",
            "reason": "toxic language",
            "scores": {"toxicity": toxic_score}
        }

    # Hate speech
    if hate_label == "hate" and hate_score > 0.8:
        return {
            "decision": "block",
            "severity": "high",
            "reason": "hate speech",
            "scores": {
                "hate": hate_score,
                "toxicity": toxic_score
            }
        }

    # Medium harmful → BLOCK (mental health app)
    if toxic_score > 0.6:
        return {
            "decision": "block",
            "severity": "medium",
            "reason": "harmful tone",
            "scores": {"toxicity": toxic_score}
        }

    # Safe
    return {"decision": "allow", "severity": "none"}


# Routes
@app.get("/")
def home():
    return {"message": "AI Moderation API is running 🚀"}


@app.post("/moderate")
def moderate(data: TextInput):
    return smart_moderation(data.text)