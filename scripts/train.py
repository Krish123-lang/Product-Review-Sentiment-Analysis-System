
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "reviews.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA).dropna()
X_train, X_test, y_train, y_test = train_test_split(
    df["review"], df["sentiment"], test_size=0.20, random_state=42, stratify=df["sentiment"]
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        max_features=5000
    )),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ))
])

pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)

report = classification_report(y_test, pred, output_dict=True, zero_division=0)
cm = confusion_matrix(y_test, pred, labels=["negative", "neutral", "positive"])

metadata = {
    "model": "TF-IDF + Logistic Regression",
    "classes": list(pipeline.classes_),
    "dataset_rows": len(df),
    "test_rows": len(X_test),
    "accuracy": round(float(accuracy_score(y_test, pred)), 4),
    "classification_report": report,
    "confusion_matrix": cm.tolist()
}

joblib.dump(pipeline, MODEL_DIR / "sentiment_pipeline.joblib")
(MODEL_DIR / "metrics.json").write_text(json.dumps(metadata, indent=2))

print(json.dumps(metadata, indent=2))
