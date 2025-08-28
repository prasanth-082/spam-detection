from flask import Flask, request, jsonify, send_from_directory
from transformers import pipeline
import os

app = Flask(__name__)

# Load a reliable public spam detection model
classifier = pipeline(
    "text-classification",
    model="mariagrandury/distilbert-base-uncased-finetuned-sms-spam-detection",
    device=-1
)

@app.route("/")
def home():
    return send_from_directory(os.getcwd(), "index.html")

@app.route("/style.css")
def serve_css():
    return send_from_directory(os.getcwd(), "style.css")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    message = data.get("message", "")

    if not message.strip():
        return jsonify({"error": "Empty message"})

    result = classifier(message, truncation=True)[0]
    label = result['label'].lower()
    score = result['score']

    # Normalize output
    if label in ["spam", "label_1"]:
        prediction = "Spam"
    else:
        prediction = "Not Spam"

    return jsonify({"prediction": prediction, "confidence": round(score, 2)})

if __name__ == "__main__":
    app.run(debug=True)
