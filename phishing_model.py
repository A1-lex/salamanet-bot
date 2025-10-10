# phishing_model.py
import os
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Load sklearn model at startup ---
try:
    with open("version_sklearn.txt", "r") as f:
        SKLEARN_VERSION = f.read().strip() or "sklearn-v1"
except FileNotFoundError:
    SKLEARN_VERSION = "sklearn-v1"

vectorizer = joblib.load("model_sklearn/vectorizer.joblib")
clf = joblib.load("model_sklearn/classifier.joblib")

# --- Load multilingual transformer (fine-tuned or fallback) ---
try:
    with open("version_transformers.txt", "r") as f:
        TRANSFORMER_VERSION = f.read().strip() or "xlmr-v1"
except FileNotFoundError:
    TRANSFORMER_VERSION = "xlmr-v1"

TRANSFORMER_DIR = os.getenv("TRANSFORMER_DIR", f"models/{TRANSFORMER_VERSION}")

# Load transformer safely
try:
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_DIR)
    transformer_model = AutoModelForSequenceClassification.from_pretrained(
        TRANSFORMER_DIR,
        num_labels=2,
        id2label={0: "safe", 1: "phishing"},
        label2id={"safe": 0, "phishing": 1}
    )
    transformer_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer_model.to(device)
    TRANSFORMER_READY = True
    print(f"✅ Transformer {TRANSFORMER_VERSION} loaded successfully on {device}")
except Exception as e:
    print(f"⚠️ Warning: Could not load transformer model: {e}")
    TRANSFORMER_READY = False


def predict_label(text: str, backend: str = None):
    """
    Predict label using sklearn, transformers, or adaptive fallback.
    """
    if backend == "sklearn":
        return predict_sklearn(text)

    elif backend == "transformers" and TRANSFORMER_READY:
        return predict_transformers(text)

    else:
        # --- Adaptive fallback logic ---
        label, proba, version, backend_used = predict_sklearn(text)

        # If sklearn is confident enough, trust it
        if proba >= 0.75:
            return label, proba, version, backend_used

        # Otherwise, try transformer if available
        if TRANSFORMER_READY:
            t_label, t_proba, t_version, t_backend = predict_transformers(text)

            # If transformer is reasonably confident (not ~50%), prefer it
            if abs(t_proba - 0.5) > 0.1:
                return t_label, t_proba, t_version, t_backend

        # Default fallback = sklearn
        return label, proba, version, backend_used


def predict_sklearn(text: str):
    """Predict with sklearn backend."""
    X = vectorizer.transform([text])
    pred = clf.predict(X)[0]
    proba = float(clf.predict_proba(X)[0][pred])
    label = "phishing" if pred == 1 else "safe"
    return label, proba, SKLEARN_VERSION, "sklearn"


def predict_transformers(text: str):
    """Predict with transformer backend (fine-tuned model)."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        logits = transformer_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze()

    # Safe = index 0, Phishing = index 1
    proba = float(probs[1])
    label = "phishing" if proba >= 0.5 else "safe"
    return label, proba, TRANSFORMER_VERSION, "transformers"
