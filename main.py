from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os
HF_TOKEN = os.getenv("HF_TOKEN")

app = FastAPI()

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# 🔥 Hinglish abuse words
BAD_WORDS = {
    "chutiya", "madarchod", "bhosdike", "gandu", "bc", "mc",
    "randi", "harami", "loda", "lund", "chodu", "bhenchod",
    "behenchod", "gand", "chut", "bhosda", "haramzada",
    "lodu", "chutiye", "madarchode"
}

class TextInput(BaseModel):
    text: str


# 🔍 API call function
def query_model(model, text):
    url = f"https://api-inference.huggingface.co/models/{model}"
    response = requests.post(url, headers=HEADERS, json={"inputs": text})
    return response.json()


def has_custom_abuse(text: str) -> bool:
    return any(word in text.lower() for word in BAD_WORDS)


def smart_moderation(text: str):

    text_lower = text.lower()

    # 🔴 Custom abuse
    if has_custom_abuse(text):
        return {
            "decision": "block",
            "severity": "high",
            "reason": "custom abusive language"
        }

    # 🟡 Call HuggingFace models
    toxic_response = query_model("unitary/toxic-bert", text)
    hate_response = query_model("Hate-speech-CNERG/bert-base-uncased-hatexplain", text)

    try:
        toxic_score = toxic_response[0]["score"]
        hate_label = hate_response[0]["label"].lower()
        hate_score = hate_response[0]["score"]
    except:
        return {"decision": "allow", "severity": "none"}

    # 🧠 Noise filter
    if toxic_score < 0.2 and hate_score < 0.2:
        return {"decision": "allow", "severity": "none"}

    # 🔴 Strong toxicity
    if toxic_score > 0.9:
        return {
            "decision": "block",
            "severity": "high",
            "reason": "toxic language",
            "scores": {"toxicity": toxic_score}
        }

    # 🔴 Hate speech (controlled)
    if hate_label == "hate" and hate_score > 0.8 and toxic_score > 0.5:
        return {
            "decision": "block",
            "severity": "high",
            "reason": "hate speech detected",
            "scores": {
                "hate": hate_score,
                "toxicity": toxic_score
            }
        }

    # 🟠 Medium warning
    if toxic_score > 0.7:
        return {
            "decision": "warn",
            "severity": "medium",
            "reason": "harmful language",
            "scores": {"toxicity": toxic_score}
        }

    # 🟡 Soft warning
    if toxic_score > 0.4:
        return {
            "decision": "warn",
            "severity": "low",
            "reason": "mild negative tone",
            "scores": {"toxicity": toxic_score}
        }

    return {"decision": "allow", "severity": "none"}


@app.get("/")
def home():
    return {"message": "AI Moderation API is running 🚀"}


@app.post("/moderate")
def moderate(data: TextInput):
    return smart_moderation(data.text)