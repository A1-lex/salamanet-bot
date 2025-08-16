🔐 SalamaNET AI

AI-Powered Cybersecurity for Africa's Education & Public Sectors

SalamaNET AI is a smart, context-aware cybersecurity system built to detect threats, monitor anomalies, and prevent phishing attacks across public and educational institutions in Africa. Our goal is to enhance digital safety through real-time, automated, and locally-relevant AI solutions.

🚨 Problem Statement

"Increased cyberthreats due to poor cyber hygiene."

Educational and public sector systems in Africa — especially in countries like Kenya — are facing a surge in cyberattacks. Most institutions lack active monitoring tools, suffer from poor password practices, and are unaware of phishing tactics.

🌍 Why It Matters

Kenyan examples:

Public agency data breaches

Schools disrupted by ransomware

Leaked citizen data due to weak security practices

These events reflect a growing crisis that demands an affordable, intelligent, and homegrown solution.

🤖 The Solution: SalamaNET AI

SalamaNET AI is an AI-based platform designed to:

🛡️ Detect phishing attempts in real time

📡 Learn continuously from user feedback

✉️ Empower institutions with actionable visibility into threats

⚙️ How It Works (MVP / Phase 1)

Right now, the MVP (Minimum Viable Product) focuses on phishing detection and learning:

User enters message → Model predicts "safe" or "phishing" + confidence
                   ↘ User feedback (correct/incorrect)
                     ↘ Feedback logged & used for retraining

Key Features:

✅ Message classification (phishing/safe + confidence score)

✅ User feedback system (mark predictions as correct/incorrect)

✅ Retraining loop (model improves once enough new feedback is collected)

✅ Admin log viewer (daily summaries, filter by model version, view history)

📂 Project Structure
salamanet_ai/
│── app.py                 # Flask web app
│── retrain_if_needed.py   # CLI helper for retraining
│── train_sklearn.py       # Training script (TF-IDF + Logistic Regression)
│── predictions_log.csv    # Saved predictions
│── feedback_log.csv       # User feedback
│── model_sklearn/         # Saved vectorizer + classifier
│── model_version.txt      # Tracks current model version
│── last_trained_count.txt # Tracks feedback count since last training
│── templates/
    └── admin_predictions.html  # Admin log viewer

🚀 Getting Started
1. Clone & Install
git clone https://github.com/A1-lex/salamanet-bot/edit/phase1-mvp
cd salamanet_ai
pip install -r requirements.txt

2. Run the App
python app.py


Access at: http://127.0.0.1:5000/

3. Retrain the Model
python retrain_if_needed.py


This script checks if there’s enough new feedback before retraining.

📊 Admin Panel

Visit: /admin/predictions

See full predictions log

Daily summaries of phishing vs safe

Filter by model version

🧠 Current Limitations (MVP Scope)

Language support: English only (Phase 2 will add Swahili, Arabic, etc.)

No explanations yet for why a message is “safe” or “phishing” (planned for Phase 2)

Focused only on phishing detection (wider cybersecurity monitoring will come later)

📅 Roadmap
Phase 1 (MVP) – ✅

Basic phishing detection (TF-IDF + Logistic Regression)

Feedback-driven learning & retraining

Admin dashboard

Phase 2 – 🔜

Explanations for predictions (“why this was flagged”)

Multilingual support (Swahili, Arabic, …)

Improved models (transformers, multilingual BERT)

Source/domain context checking

Phase 3 – 🌍

Full cybersecurity monitoring (traffic analysis, anomaly detection)

Real-time alerts & integrations

Deployment at scale

👨‍💻 Authors

SalamaNET AI is built by Alex Maina and contributors at JHUB Africa.
