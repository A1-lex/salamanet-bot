# prep_hf_dataset.py
import pandas as pd

SRC = "feedback_log.csv"
OUT = "data/feedback_multilingual.csv"

df = pd.read_csv(SRC, header=None, names=["message","label","feedback","model_version"])
df = df[df.feedback.isin(["correct","incorrect"])].copy()

def flip(row):
    if row["feedback"] == "incorrect":
        return "safe" if row["label"] == "phishing" else "phishing"
    return row["label"]

df["true_label"] = df.apply(flip, axis=1)
df["label_id"] = df.true_label.map({"safe":0,"phishing":1})
df["lang"] = df["message"].apply(lambda x: __import__("lang_utils").detect_lang(x))  # lazy import trick

df = df[["message","label_id","lang"]]
df.to_csv(OUT, index=False)
print(f"✅ Wrote {len(df)} rows to {OUT}")
