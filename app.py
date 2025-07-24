from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        message = request.form['message']
        result = classify_message(message)  # placeholder function
    return render_template('index.html', result=result)

def classify_message(text):
    # TODO: replace with AI later
    if "bank" in text.lower() or "click here" in text.lower():
        return "🚨 Suspicious Message Detected"
    return "✅ Looks Safe"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
