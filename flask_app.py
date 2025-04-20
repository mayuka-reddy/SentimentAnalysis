from flask import Flask, request, jsonify
import pickle
import numpy as np

import re

def handle_negation(text):
    # Replaces "not good" with "not_good", etc.
    return re.sub(r"\bnot\s+(\w+)", r"not_\1", text)


app = Flask(__name__)

# Load the model and vectorizer
with open("sentiment_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)
with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

@app.route("/predict", methods=["POST"])
def predict_sentiment():
    data = request.get_json()
    text = data.get("text")

    if text is None:
        return jsonify({"error": "Please provide 'text'"}), 400

    # Vectorize the text and make prediction
    processed_text = handle_negation(text)
    features = vectorizer.transform([processed_text])
    prediction = model.predict(features)[0]

    # Map the prediction output to the respective sentiment
    if prediction == 'positive':
        sentiment = "Positive"
    elif prediction == 'negative':
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return jsonify({
        "input": {"text": text},
        "predicted_sentiment": sentiment
    })

if __name__ == "__main__":
    app.run(debug=True)
