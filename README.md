# ReviewSense AI

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

### Start Streamlit

```bash
streamlit run app.py
```

Open the URL printed by Streamlit.

## 2. Run Flask API separately

In another terminal:

```bash
python api.py
```

Test health:

```bash
curl http://127.0.0.1:5000/health
```

Test prediction:

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"text\":\"The product is excellent and reliable.\"}"
```

Enable **Use Flask REST API** in the Streamlit sidebar.

## 3. Deploy to Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload this project.
3. Make sure `app.py`, `requirements.txt`, `data/`, `models/`, `services/`, and `scripts/` are committed.
4. Open Streamlit Community Cloud.
5. Create a new app.
6. Select your repository and branch.
7. Set the main file to `app.py`.
8. Deploy.

## 4. API response example

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