import os
from flask import Flask, request, render_template
from phishing_model import predict_label

app = Flask(__name__)

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     result = None
#     if request.method == 'POST':
#         message = request.form['message']
#         result = classify_message(message)  # placeholder function
#     return render_template('index.html', result=result)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        message = request.form['message'].strip()
        if message:
            result = classify_message(message)
        else:
            result = "⚠️ Please enter a message first."
    return render_template('index.html', result=result)



def classify_message(text):
    label, score = predict_label(text)

    if label.lower() in ['spam', 'phishing']:
        return f"🚨 Suspicious Message Detected ({score:.2%} confidence)"
    else:
        return f"✅ Looks Safe ({score:.2%} confidence)"


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
