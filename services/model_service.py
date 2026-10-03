from pathlib import Path
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "sentiment_pipeline.joblib"

_model = None

def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

def predict_texts(model, texts):
    probabilities = model.predict_proba(texts)
    labels = model.classes_
    results = []

    for text, probs in zip(texts, probabilities):
        idx = int(np.argmax(probs))
        sentiment = str(labels[idx])
        prob_dict = {str(label): float(prob) for label, prob in zip(labels, probs)}
        results.append((sentiment, prob_dict, float(probs[idx])))

    return results

def explain_prediction(model, text, predicted_class, top_n=8):
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]

    matrix = vectorizer.transform([text])
    feature_names = np.array(vectorizer.get_feature_names_out())

    class_index = list(classifier.classes_).index(predicted_class)
    coefficients = classifier.coef_[class_index]
    contributions = matrix.toarray()[0] * coefficients

    active = np.where(matrix.toarray()[0] != 0)[0]
    ranked = sorted(active, key=lambda i: abs(contributions[i]), reverse=True)

    output = []
    for i in ranked[:top_n]:
        output.append((feature_names[i], round(float(contributions[i]), 4)))
    return output
