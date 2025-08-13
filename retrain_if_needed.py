import pandas as pd
import subprocess
import re
import sys

# Settings
THRESHOLD = 5  # minimum new feedback items (correct + incorrect)
MODEL_VERSION_FILE = "model_version.txt"
LAST_COUNT_FILE = "last_trained_count.txt"

def get_last_trained_count():
    try:
        with open(LAST_COUNT_FILE, "r") as f:
            return int(f.read().strip())
    except FileNotFoundError:
        return 0

def set_last_trained_count(count):
    with open(LAST_COUNT_FILE, "w") as f:
        f.write(str(count))

def get_current_version():
    try:
        with open(MODEL_VERSION_FILE, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "sklearn-v0"

def bump_version(version_str):
    match = re.match(r"(sklearn-v)(\d+)", version_str)
    if match:
        prefix, num = match.groups()
        return f"{prefix}{int(num) + 1}"
    else:
        return "sklearn-v1"  # default if format unknown

def update_model_version():
    current_version = get_current_version()
    new_version = bump_version(current_version)
    with open(MODEL_VERSION_FILE, "w") as f:
        f.write(new_version)
    print(f"🔄 Updated model version to: {new_version}")
    return new_version

if __name__ == "__main__":
    try:
        df = pd.read_csv("feedback_log.csv", header=None, names=["message", "label", "feedback", "model_version"])
    except FileNotFoundError:
        print("⚠️ No feedback_log.csv found.")
        sys.exit(1)

    # ✅ Only count valid feedback ("correct" or "incorrect")
    valid_df = df[df["feedback"].isin(["correct", "incorrect"])]
    total_feedback_count = valid_df.shape[0]

    last_count = get_last_trained_count()
    new_data_count = total_feedback_count - last_count

    print(f"🧠 New feedback since last training: {new_data_count}")

    if new_data_count >= THRESHOLD:
        print("🚀 Enough data! Retraining model...")

        # Run training
        result = subprocess.run(["python", "train_sklearn.py"], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"❌ Training failed:\n{result.stderr}")
            sys.exit(1)

        # Only update version & counter if training succeeded
        set_last_trained_count(total_feedback_count)
        update_model_version()
        print("✅ Retraining complete.")
    else:
        print(f"ℹ️ Not enough new data yet. ({new_data_count}/{THRESHOLD})")
