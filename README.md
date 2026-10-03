# Product Review Sentiment Analysis System

Product Review Sentiment Analysis System built with:

- Python
- scikit-learn
- Flask REST API
- Streamlit
- TF-IDF + Logistic Regression
- Batch CSV analysis
- Confidence scores
- Basic model explainability

## Architecture

```text
                    ┌────────────────────────┐
                    │      Streamlit UI      │
                    │ single + batch + stats │
                    └───────────┬────────────┘
                                │
                    local test client OR HTTP
                                │
                    ┌───────────▼────────────┐
                    │       Flask API        │
                    │ /health   /predict     │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │  ML Pipeline           │
                    │ TF-IDF → Logistic Reg. │
                    └────────────────────────┘
```

## 1. Run locally

### Create environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### Install

```bash
pip install -r requirements.txt
```

### Train

```bash
python scripts/train.py
```

This creates:

```text
models/sentiment_pipeline.joblib
models/metrics.json
```

### Start Flask API

In one terminal:

```bash
python api.py
```

### Start Streamlit

In another terminal:

```bash
streamlit run app.py
```
Test health:

```bash
curl http://127.0.0.1:5000/health
```

Test prediction:

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"text\":\"The product is excellent and reliable.\"}"
```

## 3. API response example

`POST /predict`

```json
{
  "sentiment": "positive",
  "confidence": 0.91,
  "probabilities": {
    "negative": 0.02,
    "neutral": 0.07,
    "positive": 0.91
  }
}
```
