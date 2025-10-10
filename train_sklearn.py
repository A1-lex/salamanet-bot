# train_sklearn.py
# train_sklearn.py
import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

VERSION_FILE = "version_sklearn.txt"

# --- Load and clean feedback log ---
df = pd.read_csv(
    "feedback_log.csv",
    header=None,
    names=["message","label","feedback","model_version","lang","backend"],
    on_bad_lines="skip"
)
df = df[df.feedback.isin(["correct", "incorrect"])].copy()

# --- Flip labels for incorrect feedback ---
def flip_label(row):
    if row["feedback"] == "incorrect":
        return "safe" if row["label"] == "phishing" else "phishing"
    return row["label"]

df["true_label"] = df.apply(flip_label, axis=1)

if df.empty:
    print("⚠️ No usable feedback to train from. Exiting...")
    exit(2)   # <-- exit with non-zero to signal retrain_if_needed


# --- Encode labels ---
df['label_id'] = df.true_label.map({"safe": 0, "phishing": 1})

# --- Split train/test ---
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# --- Train TF-IDF + Logistic Regression ---
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1)
X_train = vectorizer.fit_transform(train_df.message)
y_train = train_df.label_id

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# --- Evaluate ---
X_test = vectorizer.transform(test_df.message)
acc = clf.score(X_test, test_df.label_id)
print(f"✅ SK-LR test accuracy: {acc:.2%}")

# --- Save artifacts ---
os.makedirs("model_sklearn", exist_ok=True)
joblib.dump(vectorizer, "model_sklearn/vectorizer.joblib")
joblib.dump(clf,       "model_sklearn/classifier.joblib")
print("✅ Saved TF-IDF + LR model to ./model_sklearn/")

# --- Handle version bump ---
def get_next_version(file=VERSION_FILE, base="sklearn-v"):
    try:
        with open(file, "r") as f:
            current = f.read().strip()
        match = re.match(rf"{base}(\d+)", current)
        if match:
            return f"{base}{int(match.group(1)) + 1}"
    except FileNotFoundError:
        pass
    return f"{base}1"

new_version = get_next_version()
with open(VERSION_FILE, "w") as f:
    f.write(new_version)

print(f"📌 Updated {VERSION_FILE} → {new_version}")
