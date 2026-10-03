import os
import io
import json
import pandas as pd
import streamlit as st

from services.model_service import get_model, predict_texts, explain_prediction
from services.api_client import predict_via_api, predict_via_local_flask

st.set_page_config(
    page_title="Product Review Sentiment Analysis System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
.hero {
    padding: 1.5rem 1.7rem;
    border: 1px solid rgba(128,128,128,.25);
    border-radius: 18px;
    margin-bottom: 1.3rem;
}
.small {opacity: .75; font-size: .9rem;}
.result {
    padding: 1rem;
    border-radius: 14px;
    border: 1px solid rgba(128,128,128,.25);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>Product Review Sentiment Analysis System</h1>
<p>Product review sentiment analysis using TF-IDF, Logistic Regression, Flask REST API, and Streamlit.</p>
<p class="small">Classifies reviews as Positive, Neutral, or Negative and exposes prediction confidence and basic model explainability.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("System")
    api_mode = st.toggle(
        "Use Flask REST API",
        value=bool(os.getenv("FLASK_API_URL")),
        help="If enabled, Streamlit sends predictions to the Flask API. Otherwise it uses the Flask app locally through its test client."
    )
    api_url = os.getenv("FLASK_API_URL", "http://127.0.0.1:5000")
    if api_mode:
        st.caption(f"API endpoint: {api_url}")
    st.divider()
    # st.caption("Stack")
    # st.write("Python • scikit-learn • Flask • Streamlit")
    # st.caption("Model")
    # st.write("TF-IDF + Logistic Regression")

model = get_model()

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["Single Review", "Batch Analysis", "Model Insights"])

with tab1:
    st.subheader("Analyze a review")
    examples = [
        "The battery life is excellent and the build quality feels premium.",
        "It works, but nothing really stands out for the price.",
        "The product stopped working after three days and feels cheaply made.",
    ]
    example = st.selectbox("Example", ["Write my own"] + examples)
    default_text = "" if example == "Write my own" else example
    review = st.text_area(
        "Review text",
        value=default_text,
        height=150,
        placeholder="Example: The product is fast, reliable, and worth the money.",
    )

    if st.button("Analyze sentiment", type="primary", use_container_width=True):
        if not review.strip():
            st.warning("Enter a review first.")
        else:
            try:
                if api_mode:
                    result = predict_via_api(review, api_url)
                else:
                    result = predict_via_local_flask(review)
                predicted = result["sentiment"]
                probs = result["probabilities"]
                confidence = result["confidence"]

                label_icons = {"positive": "Positive", "neutral": "Neutral", "negative": "Negative"}
                st.markdown(f"### Prediction: **{label_icons[predicted]}**")
                st.progress(float(confidence), text=f"Confidence: {confidence:.1%}")

                c1, c2, c3 = st.columns(3)
                for col, label in zip((c1, c2, c3), ["positive", "neutral", "negative"]):
                    col.metric(label.title(), f"{probs.get(label, 0):.1%}")

                st.divider()
                st.subheader("Why did the model lean this way?")
                terms = explain_prediction(model, review, predicted, top_n=8)
                if terms:
                    st.dataframe(
                        pd.DataFrame(terms, columns=["Term", "Contribution"]),
                        hide_index=True,
                        use_container_width=True,
                    )
                else:
                    st.info("No informative terms were extracted from this review.")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

with tab2:
    st.subheader("Analyze many reviews")
    uploaded = st.file_uploader("Upload a CSV containing a `review` column", type=["csv"])

    if uploaded:
        batch = pd.read_csv(uploaded)
        if "review" not in batch.columns:
            st.error("The CSV must contain a column named `review`.")
        else:
            batch["review"] = batch["review"].fillna("").astype(str)
            if st.button("Run batch analysis", type="primary"):
                try:
                    if api_mode:
                        results = [predict_via_api(text, api_url) for text in batch["review"]]
                        batch["sentiment"] = [r["sentiment"] for r in results]
                        batch["confidence"] = [r["confidence"] for r in results]
                    else:
                        predictions = predict_texts(model, batch["review"].tolist())
                        batch["sentiment"] = [p[0] for p in predictions]
                        batch["confidence"] = [p[2] for p in predictions]

                    st.success(f"Analyzed {len(batch):,} reviews.")
                    st.dataframe(batch, use_container_width=True, hide_index=True)

                    counts = batch["sentiment"].value_counts()
                    st.bar_chart(counts)

                    csv_bytes = batch.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download analyzed CSV",
                        csv_bytes,
                        file_name="sentiment_results.csv",
                        mime="text/csv",
                    )
                except Exception as exc:
                    st.error(f"Batch analysis failed: {exc}")
    else:
        st.info("Upload a CSV to analyze multiple reviews at once.")

with tab3:
    st.subheader("Model performance")
    metrics_path = "models/metrics.json"
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        c1, c2, c3 = st.columns(3)
        c1.metric("Test accuracy", f"{metrics['accuracy']:.1%}")
        c2.metric("Training rows", f"{metrics['dataset_rows']:,}")
        c3.metric("Test rows", f"{metrics['test_rows']:,}")

        st.write("Classification report")
        report = pd.DataFrame(metrics["classification_report"]).T
        st.dataframe(report.round(3), use_container_width=True)

        st.write("Confusion matrix")
        cm = pd.DataFrame(
            metrics["confusion_matrix"],
            index=["Actual negative", "Actual neutral", "Actual positive"],
            columns=["Pred. negative", "Pred. neutral", "Pred. positive"],
        )
        st.dataframe(cm, use_container_width=True)
    except FileNotFoundError:
        st.warning("Metrics file not found. Run `python scripts/train.py` first.")

    st.divider()
#     st.subheader("How the system works")
#     st.markdown("""
# 1. **Text input** is validated and normalized.
# 2. **TF-IDF** converts the review into numerical features using unigrams and bigrams.
# 3. **Logistic Regression** predicts Positive, Neutral, or Negative.
# 4. **Probability scores** provide a confidence estimate.
# 5. **Flask** exposes the same model through `/health` and `/predict`.
# 6. **Streamlit** provides the interactive UI and batch-analysis workflow.
# """)
