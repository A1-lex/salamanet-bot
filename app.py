from flask import Flask, request, render_template, redirect, url_for
import csv
from phishing_model import predict_label
import subprocess
from datetime import datetime

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    message = ""
    label = ""
    model_version = "unknown"

    if request.method == 'POST':
        if 'feedback' in request.form:
            feedback = request.form['feedback']
            message = request.form['message']
            label = request.form['label']
            model_version = request.form.get('model_version', 'unknown')
            log_feedback(message, label, feedback, model_version)
            result = f"✅ Thanks! Feedback recorded as '{feedback}'"
        else:
            message = request.form['message'].strip()
            if message:
                label, result, model_version = classify_message(message)
                log_prediction(message, label, model_version, result)
            else:
                result = "⚠️ Please enter a message first."

    return render_template(
        'index.html',
        result=result,
        message=message,
        label=label,
        model_version=model_version
    )


def classify_message(text):
    label, score, version = predict_label(text)
    if label.lower() in ['spam', 'phishing']:
        status = "phishing"
        display = f"🚨 Suspicious Message Detected ({score:.2%} confidence)"
    else:
        status = "safe"
        display = f"✅ Looks Safe ({score:.2%} confidence)"
    return status, display, version


def log_feedback(message, label, feedback, model_version):
    with open("feedback_log.csv", "a", encoding="utf-8") as f:
        f.write(f'"{message}","{label}","{feedback}","{model_version}"\n')


def log_prediction(message, label, model_version, display_text):
    """
    Logs a prediction to predictions_log.csv with strict formatting to prevent parsing errors.
    """
    # Extract numeric confidence safely
    try:
        confidence_str = display_text.split("(")[1].split("%")[0].strip()
        confidence = round(float(confidence_str) / 100, 4)  # Always store as 0.xxxx
    except Exception:
        confidence = 0.0

    # Clean commas/newlines to keep CSV structure intact
    clean_message = message.replace('"', '""').replace("\n", " ").replace("\r", " ")
    clean_label = label.strip()
    clean_model_version = model_version.strip()

    # Timestamp in consistent format
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Write in strict CSV format
    with open("predictions_log.csv", "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow([clean_message, clean_label, confidence, clean_model_version, timestamp])



@app.route('/admin/logs')
def view_logs():
    log_entries = []
    version_filter = request.args.get("model_version", "").strip()

    try:
        with open("feedback_log.csv", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 4:
                    entry = {
                        "message": row[0],
                        "label": row[1],
                        "feedback": row[2],
                        "model_version": row[3]
                    }
                    if not version_filter or version_filter.lower() == "any":
                        log_entries.append(entry)
                    elif entry["model_version"] == version_filter:
                        log_entries.append(entry)
    except FileNotFoundError:
        return "<h2>No feedback logs found.</h2>"

    correct = sum(1 for entry in log_entries if entry["feedback"] == "correct")
    incorrect = sum(1 for entry in log_entries if entry["feedback"] == "incorrect")
    total = correct + incorrect
    accuracy = (correct / total * 100) if total else 0.0

    return render_template(
        "admin_logs.html",
        logs=log_entries,
        version=version_filter,
        selected_version=version_filter if version_filter else None,
        correct=correct,
        incorrect=incorrect,
        accuracy=accuracy
    )


@app.route('/admin/predictions')
def view_predictions():
    log_entries = []
    version_filter = request.args.get("version", "").strip()

    try:
        with open("predictions_log.csv", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                try:
                    if len(row) == 5:
                        confidence = float(row[2])  # may throw ValueError
                        entry = {
                            "message": row[0],
                            "label": row[1],
                            "confidence": confidence,
                            "model_version": row[3],
                            "timestamp": row[4]
                        }
                        if not version_filter or version_filter.lower() == "any":
                            log_entries.append(entry)
                        elif entry["model_version"] == version_filter:
                            log_entries.append(entry)
                except ValueError:
                    # Skip bad/malformed rows silently
                    continue
    except FileNotFoundError:
        return "<h2>No predictions logs found.</h2>"

    # --- ✅ Daily Summary ---
    from collections import defaultdict
    daily_summary = defaultdict(lambda: {"total": 0, "phishing": 0, "safe": 0})
    for entry in log_entries:
        day = entry["timestamp"].split(" ")[0]  # YYYY-MM-DD
        daily_summary[day]["total"] += 1
        if entry["label"] == "phishing":
            daily_summary[day]["phishing"] += 1
        else:
            daily_summary[day]["safe"] += 1

    daily_summary_list = [
        {"date": day, "total": stats["total"], "phishing": stats["phishing"], "safe": stats["safe"]}
        for day, stats in sorted(daily_summary.items(), reverse=True)
    ]

    return render_template(
        "admin_predictions.html",
        logs=log_entries,
        daily_summary=daily_summary_list,
        version=version_filter,
        selected_version=version_filter if version_filter else None
    )



@app.route("/admin/retrain", methods=["POST"])
def retrain_now():
    try:
        result = subprocess.run(["python", "retrain_if_needed.py"], capture_output=True, text=True)
        output = result.stdout + "\n" + result.stderr
        return f"<pre>{output}</pre><br><a href='{url_for('view_logs')}'>⬅ Back to Logs</a>"
    except Exception as e:
        return f"❌ Error during retraining: {e}"


if __name__ == '__main__':
    app.run(debug=True)
