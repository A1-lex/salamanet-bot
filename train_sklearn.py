import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# 1. Load and clean feedback log
df = pd.read_csv("feedback_log.csv", header=None, names=["message","label","feedback","model_version"])

# 2. Keep only rows with feedback we can use
df = df[df.feedback.isin(["correct", "incorrect"])].copy()

# 3. Flip labels for incorrect feedback
def flip_label(row):
    if row["feedback"] == "incorrect":
        return "safe" if row["label"] == "phishing" else "phishing"
    return row["label"]

df["true_label"] = df.apply(flip_label, axis=1)

# 4. Check if we have enough data
if df.empty:
    print("⚠️ No usable feedback to train from. Exiting...")
    exit()

# 5. Encode labels
df['label_id'] = df.true_label.map({"safe": 0, "phishing": 1})

# 6. Split train/test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 7. Train TF-IDF + Logistic Regression
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1)
X_train = vectorizer.fit_transform(train_df.message)
y_train = train_df.label_id

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# 8. Evaluate on hold-out set
X_test = vectorizer.transform(test_df.message)
acc = clf.score(X_test, test_df.label_id)
print(f"✅ SK-LR test accuracy: {acc:.2%}")

# 9. Save artifacts
os.makedirs("model_sklearn", exist_ok=True)
joblib.dump(vectorizer, "model_sklearn/vectorizer.joblib")
joblib.dump(clf,       "model_sklearn/classifier.joblib")
print("✅ Saved TF-IDF + LR model to ./model_sklearn/")
