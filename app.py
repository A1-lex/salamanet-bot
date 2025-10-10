from flask import Flask, request, render_template, url_for
import csv
import subprocess
from datetime import datetime
from phishing_model import predict_label
from lang_utils import detect_lang

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    message = ""
    label = ""
    model_version = "unknown"
    confidence = None
    backend = "unknown"
    lang = None

    if request.method == 'POST':
        # ✅ Feedback handling
        if 'feedback' in request.form:
            feedback = request.form['feedback']
            message = request.form['message']
            label = request.form['label']
            model_version = request.form.get('model_version', 'unknown')
            lang = request.form.get('lang', 'unknown')
            backend = request.form.get('backend', 'unknown')

            # log feedback as-is (don’t recompute backend here)
            log_feedback(message, label, feedback, model_version, lang, backend)
            result = f"✅ Thanks! Feedback recorded as '{feedback}'"

        # ✅ Prediction handling
        else:
            message = request.form['message'].strip()
            if message:
                # Now returns confidence too
                status, display, version, backend, confidence = classify_message(message)
                lang = detect_lang(message)

                log_prediction(message, status, version, display, lang, backend, confidence)
                label = status
                model_version = version
                result = display
            else:
                result = "⚠️ Please enter a message first."

    return render_template(
        'index.html',
        result=result,
        message=message,
        label=label,
        model_version=model_version,
        backend=backend,
        confidence=confidence if confidence is not None else 0.0,
        lang=lang
    )


def classify_message(text):
    """Run prediction and format result for display (backend-agnostic)."""
    label, score, version, backend = predict_label(text)

    # Handle confidence extraction
    confidence = None
    if score is not None:
        confidence = float(score)

    # Standardize status & display
    if label.lower() in ['spam', 'phishing']:
        status = "phishing"
        display = f"🚨 Suspicious Message Detected ({confidence:.2%} confidence)" if confidence else "🚨 Suspicious Message Detected"
    else:
        status = "safe"
        display = f"✅ Looks Safe ({confidence:.2%} confidence)" if confidence else "✅ Looks Safe"

    return status, display, version, backend, confidence


def log_feedback(message, label, feedback, model_version, lang, backend):
    """Store user feedback in feedback_log.csv with lang + backend (as used)."""
    with open("feedback_log.csv", "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow([message, label, feedback, model_version, lang, backend])


def log_prediction(message, label, model_version, display_text, lang, backend, confidence):
    """Log predictions with full metadata to predictions_log.csv"""
    clean_message = message.replace('"', '""').replace("\n", " ").replace("\r", " ")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open("predictions_log.csv", "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow([clean_message, label, confidence, model_version, timestamp, lang, backend])


@app.route('/admin/logs')
def view_logs():
    """Admin view for feedback logs (handles old and new formats)."""
    log_entries = []
    version_filter = request.args.get("model_version", "").strip()

    try:
        with open("feedback_log.csv", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 4:  # old format
                    msg, lab, fb, ver = row
                    lang = "unknown"
                    backend = "sklearn"
                elif len(row) == 6:  # new format
                    msg, lab, fb, ver, lang, backend = row
                else:
                    continue  # skip malformed rows

                entry = {
                    "message": msg,
                    "label": lab,
                    "feedback": fb,
                    "model_version": ver,
                    "lang": lang,
                    "backend": backend
                }

                # Apply filter
                if not version_filter or version_filter.lower() == "any":
                    log_entries.append(entry)
                elif entry["model_version"] == version_filter:
                    log_entries.append(entry)
    except FileNotFoundError:
        return "<h2>No feedback logs found.</h2>"

    # Accuracy calc
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
    """Admin view for predictions logs"""
    log_entries = []
    version_filter = request.args.get("version", "").strip()
    lang_filter = request.args.get("lang", "").strip().lower()
    backend_filter = request.args.get("backend", "").strip().lower()

    try:
        with open("predictions_log.csv", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 5:  # old format
                    msg, lab, conf, ver, ts = row
                    lang = "unknown"
                    backend = "sklearn"
                elif len(row) == 7:  # new format
                    msg, lab, conf, ver, ts, lang, backend = row
                else:
                    continue

                try:
                    conf = float(conf)
                except:
                    continue

                entry = {
                    "message": msg,
                    "label": lab,
                    "confidence": conf,
                    "model_version": ver,
                    "timestamp": ts,
                    "lang": lang,
                    "backend": backend
                }

                if version_filter and version_filter.lower() != "any" and ver != version_filter:
                    continue
                if lang_filter and lang_filter != "any" and entry["lang"].lower() != lang_filter:
                    continue
                if backend_filter and backend_filter != "any" and entry["backend"].lower() != backend_filter:
                    continue

                log_entries.append(entry)
    except FileNotFoundError:
        return "<h2>No predictions logs found.</h2>"

    from collections import defaultdict
    daily_summary = defaultdict(lambda: {"total": 0, "phishing": 0, "safe": 0})
    for e in log_entries:
        day = e["timestamp"].split(" ")[0]
        daily_summary[day]["total"] += 1
        daily_summary[day]["phishing" if e["label"] == "phishing" else "safe"] += 1

    daily_summary_list = [
        {"date": day, "total": s["total"], "phishing": s["phishing"], "safe": s["safe"]}
        for day, s in sorted(daily_summary.items(), reverse=True)
    ]

    return render_template(
        "admin_predictions.html",
        logs=log_entries,
        daily_summary=daily_summary_list,
        version=version_filter,
        lang=lang_filter,
        backend=backend_filter,
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
