# train_transformer.py
import os, re, pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

DATA = "data/feedback_multilingual.csv"
MODEL_NAME = "xlm-roberta-base"
BASE_DIR = "models"

# --- Load data ---
df = pd.read_csv(DATA)
if len(df) < 20:
    print("⚠️ Not enough data for transformer prototype (need ~20+).")
    raise SystemExit(0)

print("Label distribution:\n", df["label_id"].value_counts())

dataset = Dataset.from_pandas(df[["message", "label_id", "lang"]])

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tok(batch):
    return tokenizer(
        batch["message"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

dataset = dataset.map(tok, batched=True)
dataset = dataset.rename_column("label_id", "labels")
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_ds, eval_ds = dataset["train"], dataset["test"]

# --- Auto-increment transformer version ---
def get_next_version(base="xlmr"):
    existing = [d for d in os.listdir(BASE_DIR) if d.startswith(base)]
    if not existing:
        return f"{base}-v1"
    nums = [int(re.search(r"v(\d+)", d).group(1)) for d in existing if re.search(r"v(\d+)", d)]
    return f"{base}-v{max(nums) + 1}"

OUT_VERSION = get_next_version()
OUT_DIR = os.path.join(BASE_DIR, OUT_VERSION)

# --- Model ---
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
)

# --- Training args ---
args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=int(os.getenv("BATCH_SIZE", 8)),
    per_device_eval_batch_size=8,
    num_train_epochs=int(os.getenv("EPOCHS", 3)),
    learning_rate=float(os.getenv("LR", 5e-5)),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True
)

# --- Metrics ---
def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids
    acc = (preds == labels).mean()
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1)
    }

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

# --- Save transformer version ---
with open("version_transformers.txt", "w") as f:
    f.write(OUT_VERSION)

print(f"✅ Saved transformer model to {OUT_DIR}")
print(f"📌 Updated version_transformers.txt → {OUT_VERSION}")
