from flask import Flask, jsonify, request
from services.model_service import get_model, predict_texts

app = Flask(__name__)
model = get_model()

@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "ReviewSense AI",
        "model": "TF-IDF + Logistic Regression"
    })

@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text")

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "JSON body must contain a non-empty string field named 'text'."}), 400

    sentiment, probabilities, confidence = predict_texts(model, [text])[0]
    return jsonify({
        "sentiment": sentiment,
        "confidence": round(confidence, 6),
        "probabilities": {k: round(v, 6) for k, v in probabilities.items()}
    })

@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Endpoint not found."}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
