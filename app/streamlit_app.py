"""Resume Analyzer demo.

    streamlit run app/streamlit_app.py

Shows every trained model found in runs/ and models/ (see src/registry.py).
"""
import html
import io
import os
import sys

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import registry, skills  # noqa: E402
from src.dl.text import clean, strip_title, tokenize  # noqa: E402

st.set_page_config(page_title="Resume Analyzer", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def load_model(name):
    cls, path = registry.discover(ROOT)[name]
    return cls(path)


def read_pdf(data):
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def highlight(text, words, limit=3000):
    """Mark the model's most important words inside the resume text."""
    if not words:
        return html.escape(text[:limit])
    top = max(w for _, w in words) or 1.0
    weight = {w.lower(): s / top for w, s in words}
    out = []
    for token in text[:limit].split(" "):
        key = token.lower().strip(".,;:()[]\"'")
        if key in weight:
            alpha = 0.25 + 0.6 * weight[key]
            out.append(f'<mark style="background: rgba(255,193,7,{alpha:.2f}); padding: 0 2px; border-radius: 3px;">'
                       f"{html.escape(token)}</mark>")
        else:
            out.append(html.escape(token))
    return " ".join(out)


st.title("Resume Analyzer")
st.caption("Predicts the job category of a resume (24 categories) and compares it with a job description. "
           "NLP course project, HCMUT.")

available = registry.discover(ROOT)
with st.sidebar:
    st.header("Model")
    if not available:
        st.error("No trained models found. Train one first, e.g.\n\n"
                 "`python -m src.dl.train --data data/raw/Resume.csv --model textcnn`\n\n"
                 "or copy the team's models into `runs/` and `models/`.")
        st.stop()
    chosen = st.selectbox("Classifier", list(available))
    compare_all = st.checkbox("Compare all models", value=False)
    drop_title = st.checkbox("Ignore the job title line", value=False,
                             help="Removes the title at the top of the resume, which often gives the category away.")
    st.divider()
    st.caption(f"{len(available)} model(s) available")

tab_analyse, tab_results = st.tabs(["Analyse a resume", "Model results"])

with tab_analyse:
    left, right = st.columns(2)
    with left:
        st.subheader("Resume")
        up = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
        pasted = st.text_area("or paste the resume text", height=220)
    with right:
        st.subheader("Job description (optional)")
        jd = st.text_area("Paste a job description to check the match", height=300)

    text = ""
    if up is not None:
        try:
            text = read_pdf(up.getvalue()) if up.name.lower().endswith(".pdf") else up.getvalue().decode("utf-8", "ignore")
        except Exception as e:  # unreadable or scanned PDF
            st.error(f"Could not read this file: {e}")
        if up.name.lower().endswith(".pdf") and not text.strip():
            st.warning("No text found in this PDF. It may be a scanned image; paste the text instead.")
    text = text or pasted

    if st.button("Analyse", type="primary"):
        if len(text.split()) < 20:
            st.warning("Please provide a resume with at least 20 words.")
            st.stop()
        model_text = strip_title(text) if drop_title else text
        model = load_model(chosen)
        preds = model.predict(model_text, k=3)

        st.divider()
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Predicted category")
            st.metric(preds[0][0], f"{preds[0][1]:.0%}", help="Model confidence for the top category")
            st.bar_chart(pd.DataFrame(preds, columns=["Category", "Probability"]).set_index("Category"),
                         horizontal=True, height=180)
            if isinstance(model, registry.SklearnModel) and not hasattr(model._pipe, "predict_proba"):
                st.caption("This model gives scores, not calibrated probabilities.")
            if preds[0][1] < 0.4:
                st.info("Low confidence: this resume may fit several categories.")
        with c2:
            st.subheader("Words behind the decision")
            words = model.top_words(model_text, k=15)
            if words:
                st.dataframe(pd.DataFrame(words, columns=["Word", "Weight"]).round(3), hide_index=True, height=250)
            else:
                st.caption("Not available for this model. Attention models (BiLSTM, CNN+BiLSTM) and linear baselines show them.")

        if compare_all and len(available) > 1:
            st.subheader("All models")
            rows = []
            for name in available:
                try:
                    m = load_model(name)
                    p = m.predict(model_text, k=3)
                    rows.append({"Model": name, "Type": m.kind, "Prediction": p[0][0], "Confidence": round(p[0][1], 3),
                                 "2nd": p[1][0], "3rd": p[2][0]})
                except Exception as e:
                    rows.append({"Model": name, "Type": "error", "Prediction": str(e)[:80]})
            st.dataframe(pd.DataFrame(rows), hide_index=True)

        if jd.strip():
            st.divider()
            st.subheader("Match with the job description")
            res = skills.match(text, jd)
            sim = skills.tfidf_similarity(text, jd)
            a, b, c = st.columns(3)
            a.metric("Required skills found", "n/a" if res["coverage"] is None else f"{res['coverage']:.0%}")
            b.metric("Skills matched / missing", f"{len(res['matched'])} / {len(res['missing'])}")
            c.metric("Text similarity (TF-IDF)", f"{sim:.2f}")
            m1, m2 = st.columns(2)
            m1.markdown("**Matched skills**\n\n" + (", ".join(res["matched"]) or "none"))
            m2.markdown("**Missing skills**\n\n" + (", ".join(res["missing"]) or "none"))
            st.caption("Skill matching uses a fixed skill list and exact wording; it is a rule-based helper, not a trained model.")

        st.divider()
        with st.expander("Resume text with the important words highlighted", expanded=bool(words)):
            st.markdown(f'<div style="line-height:1.7">{highlight(model_text, words)}</div>', unsafe_allow_html=True)
        with st.expander("What the model sees after preprocessing"):
            st.code(" ".join(tokenize(clean(model_text))[:300]), language=None)

with tab_results:
    st.subheader("Test-set results")
    path = os.path.join(ROOT, "reports", "results.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        show = [c for c in ["Model", "Representation", "Job title", "Val Macro-F1", "Test Macro-F1", "Test Accuracy",
                            "Params", "Train time (s)"] if c in df.columns]
        st.dataframe(df[show].sort_values("Test Macro-F1", ascending=False), hide_index=True)
        chart = os.path.join(ROOT, "reports", "model_comparison.png")
        if os.path.exists(chart):
            st.image(chart)
    else:
        st.info("No results yet: `reports/results.csv` is created when models are trained.")
    if chosen in available and available[chosen][0] is registry.DLModel:
        run = available[chosen][1]
        for img, cap in (("curves.png", "Training curves"), ("confusion_matrix.png", "Confusion matrix (test)")):
            if os.path.exists(os.path.join(run, img)):
                st.image(os.path.join(run, img), caption=f"{cap}: {chosen}")
