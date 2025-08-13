import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

# 1. Load and preprocess feedback_log.csv
df = pd.read_csv("feedback_log.csv", header=None, names=["message", "model_output", "feedback", "model_version"])


# 2. Extract correct feedback only
df = df[df["feedback"] == "correct"]

# 3. Label data manually
def extract_label(model_output):
    if "Suspicious" in model_output:
        return "phishing"
    else:
        return "safe"

df["label"] = df["model_output"].apply(extract_label)

# 4. Map labels to integers
label_map = {"safe": 0, "phishing": 1}
df["label_id"] = df["label"].map(label_map)

# 5. Load tokenizer and model
model_name = "bert-base-multilingual-cased"  # Lightweight multilingual model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 6. Tokenize inputs
def tokenize(example):
    return tokenizer(example["message"], truncation=True, padding="max_length", max_length=128)

# 7. Prepare Hugging Face Dataset
dataset = Dataset.from_pandas(df[["message", "label_id"]])
dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("label_id", "labels")

# 8. Split into train/test
train_dataset, eval_dataset = dataset.train_test_split(test_size=0.2).values()

# 9. Load model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 10. Training arguments
training_args = TrainingArguments(
    output_dir="./model-v1",
    evaluation_strategy="epoch",
    save_strategy="epoch",  # ✅ match evaluation
    logging_dir="./logs",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    save_total_limit=1,
    logging_steps=5,
    load_best_model_at_end=True
)


# 11. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

# 12. Train!
trainer.train()

# 13. Save final model locally
trainer.save_model("./model-v1")
tokenizer.save_pretrained("./model-v1")

# Save the current number of correct feedback entries
with open("last_training_count.txt", "w") as f:
    f.write(str(len(df)))
print(f"📦 Logged current training count: {len(df)} entries")

print("✅ Training complete. Custom model saved to ./model-v1")



