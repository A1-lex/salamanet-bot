# retrain_if_needed.py
import pandas as pd
import subprocess
import sys

# Settings
THRESHOLD = 5  # minimum new feedback items
VERSION_FILE = "version_sklearn.txt"
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

if __name__ == "__main__":
    try:
        df = pd.read_csv(
            "feedback_log.csv",
            header=None,
            names=["message","label","feedback","model_version","lang","backend"],
            on_bad_lines="skip"
        )
    except FileNotFoundError:
        print("⚠️ No feedback_log.csv found.")
        sys.exit(1)

    valid_df = df[df["feedback"].isin(["correct", "incorrect"])]
    total_feedback_count = valid_df.shape[0]

    last_count = get_last_trained_count()
    new_data_count = total_feedback_count - last_count

    print(f"🧠 New feedback since last training: {new_data_count}")

    if new_data_count >= THRESHOLD:
        print("🚀 Enough data! Retraining model...")

        result = subprocess.run(
            ["python", "train_sklearn.py"],
            capture_output=True,
            text=True
        )

        stdout, stderr = result.stdout.strip(), result.stderr.strip()

        if result.returncode == 0:
            # ✅ Successful training
            print(stdout)
            set_last_trained_count(total_feedback_count)
            print("✅ Retraining complete.")

        elif result.returncode == 2:
            # ⚠️ Graceful "no usable feedback" exit
            print(stdout or "⚠️ No usable feedback detected, training skipped.")
            print("ℹ️ Counters not updated — waiting for more/better feedback.")

        else:
            # ❌ Real failure
            print("❌ Training failed unexpectedly!")
            if stdout:
                print("STDOUT:\n", stdout)
            if stderr:
                print("STDERR:\n", stderr)
            sys.exit(result.returncode)

    else:
        print(f"ℹ️ Not enough new data yet. ({new_data_count}/{THRESHOLD})")
