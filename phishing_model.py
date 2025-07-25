# from transformers import pipeline

# # Load the multilingual phishing detection model
# classifier = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-sms-spam-detection")

# def predict_label(text):
#     result = classifier(text)[0]
#     label = result['label']
#     score = result['score']
#     return label, score

from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")

def predict_label(text):
    labels = ["phishing", "safe", "spam", "fraud", "scam"]
    result = classifier(text, candidate_labels=labels, multi_label=True)
    
    # Take the highest scoring label
    best_label = result['labels'][0]
    best_score = result['scores'][0]
    return best_label, best_score

def classify_message(text):
    suspicious_keywords = ['bonyeza hapa', 'airtel money', 'bit.ly', 'm-pesa', 'click here']
    lower = text.lower()
    if any(kw in lower for kw in suspicious_keywords):
        return "🚨 Suspicious (keyword match)"
    label, score = predict_label(text)
    if label in ['phishing', 'scam', 'fraud'] and score > 0.6:
        return f"🚨 Suspicious Message Detected ({score:.2%} confidence)"
    return f"✅ Looks Safe ({score:.2%} confidence)"
