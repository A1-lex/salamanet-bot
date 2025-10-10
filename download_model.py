from transformers import AutoTokenizer, AutoModelForSequenceClassification

def main():
    MODEL_NAME = "xlm-roberta-base"   # multilingual transformer
    SAVE_DIR = "./models/xlmr-v1"

    print("⏳ Downloading model from Hugging Face Hub:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    print(f"💾 Saving model to {SAVE_DIR} ...")
    tokenizer.save_pretrained(SAVE_DIR)
    model.save_pretrained(SAVE_DIR)

    print("✅ Done. Model saved locally.")

if __name__ == "__main__":
    main()
