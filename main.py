from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# ✅ Load models once
toxic_model = pipeline("text-classification", model="unitary/toxic-bert")
hate_model = pipeline("text-classification", model="Hate-speech-CNERG/bert-base-uncased-hatexplain")

# 🔥 Hinglish abuse words
BAD_WORDS = {
    "chutiya", "madarchod", "bhosdike", "gandu", "bc", "mc",
    "randi", "harami", "loda", "lund", "chodu", "bhenchod",
    "behenchod", "gand", "chut", "bhosda", "haramzada",
    "lodu", "chutiye", "madarchode"
}

# 🧠 Request schema
class TextInput(BaseModel):
    text: str

# 🔍 Custom abuse check
def has_custom_abuse(text: str) -> bool:
    text = text.lower()
    return any(word in text for word in BAD_WORDS)

# 🧠 Core moderation logic
def smart_moderation(text: str):

    text_lower = text.lower()

    # 🔴 1. Custom abuse → immediate block
    if has_custom_abuse(text):
        return {
            "decision": "block",
            "severity": "high",
            "reason": "custom abusive language"
        }

    # 🟡 2. Model predictions
    toxic_output = toxic_model(text)[0]
    toxic_score = float(toxic_output["score"])

    hate_output = hate_model(text)[0]
    hate_label = hate_output["label"].lower()
    hate_score = float(hate_output["score"])

    # 🧠 3. Noise filtering (VERY IMPORTANT)
    MIN_THRESHOLD = 0.2

    if toxic_score < MIN_THRESHOLD and hate_score < MIN_THRESHOLD:
        return {
            "decision": "allow",
            "severity": "none"
        }

    # 🔴 4. Strong toxicity → BLOCK
    if toxic_score > 0.9:
        return {
            "decision": "block",
            "severity": "high",
            "reason": "toxic language",
            "scores": {
                "toxicity": toxic_score
            }
        }

    # 🔴 5. Hate speech (controlled, avoid false positives)
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

    # 🔴 6. Harassment detection
    if text_lower.count("you") >= 3 and toxic_score > 0.7:
        return {
            "decision": "block",
            "severity": "high",
            "reason": "harassment detected"
        }

    # 🟠 7. Medium toxicity → WARN
    if toxic_score > 0.7:
        return {
            "decision": "warn",
            "severity": "medium",
            "reason": "potentially harmful language",
            "scores": {
                "toxicity": toxic_score
            }
        }

    # 🟡 8. Mild toxicity → SOFT WARN
    if toxic_score > 0.4:
        return {
            "decision": "warn",
            "severity": "low",
            "reason": "mild negative tone",
            "scores": {
                "toxicity": toxic_score
            }
        }

    # 🟢 9. Safe content
    return {
        "decision": "allow",
        "severity": "none"
    }

# ✅ Health route
@app.get("/")
def home():
    return {"message": "AI Moderation API is running 🚀"}

# ✅ API endpoint
@app.post("/moderate")
def moderate(data: TextInput):
    try:
        return smart_moderation(data.text)
    except Exception as e:
        return {
            "decision": "allow",  # fail-safe
            "severity": "none",
            "error": str(e)
        }