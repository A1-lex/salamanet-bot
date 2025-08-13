import joblib
from datetime import datetime
import csv

# Load version from file
with open("model_version.txt", "r") as f:
    MODEL_VERSION = f.read().strip()

# Load vectorizer and classifier
vectorizer = joblib.load("model_sklearn/vectorizer.joblib")
clf = joblib.load("model_sklearn/classifier.joblib")

def predict_label(text):
    X = vectorizer.transform([text])
    pred = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0][pred]
    label = "phishing" if pred == 1 else "safe"

    # Log prediction automatically
    log_prediction(text, label, proba, MODEL_VERSION)

    return label, proba, MODEL_VERSION


def log_prediction(message, label, confidence, model_version):
    """Append predictions to predictions_log.csv for analytics."""
    with open("predictions_log.csv", "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            message,
            label,
            f"{confidence:.4f}",
            model_version
        ])
