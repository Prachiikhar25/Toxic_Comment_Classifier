from flask import Flask, request, render_template, url_for
from src.predict import predict_text

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', result=None)


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    text = request.values.get('text', '').strip()
    if not text:
        return render_template('index.html', result=None)
    try:
        result = predict_text(text)
    except Exception as e:
        # model not found or other error
        return f"Error: {e}", 500
    # convert to ints for display
    result['labels'] = {k: int(v) for k, v in result['labels'].items()}
    return render_template('index.html', result=result)


if __name__ == '__main__':
    # debug True enables auto-reload during development
    app.run(debug=True, port=5000)
