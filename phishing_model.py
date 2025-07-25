# from transformers import pipeline

# # Load the multilingual phishing detection model
# classifier = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-sms-spam-detection")

# def predict_label(text):
#     result = classifier(text)[0]
#     label = result['label']
#     score = result['score']
#     return label, score

from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")

def predict_label(text):
    labels = ["phishing", "safe", "scam", "spam", "fraud"]
    result = classifier(text, candidate_labels=labels, multi_label=True)
    best_label = result['labels'][0]
    best_score = result['scores'][0]
    return best_label, best_score
