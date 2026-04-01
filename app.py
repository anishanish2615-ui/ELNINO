import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import numpy as np

st.set_page_config(page_title='Review Trust Analyzer', layout='centered')

# Advanced Flipkart-inspired Styling
st.markdown("""
    <style>
    .main { background-color: #f1f3f6; }
    .header-box {
        background-color: #2874f0;
        padding: 15px;
        color: white;
        text-align: left;
        padding-left: 10% !important;
        margin: -5rem -10rem 2rem -10rem;
    }
    .stTextArea textarea {
        border-radius: 4px;
        border: 1px solid #dbdbdb;
    }
    .stButton>button {
        width: 100%;
        background-color: #fb641b;
        color: white;
        font-weight: bold;
        border-radius: 2px;
        border: none;
        height: 48px;
        text-transform: uppercase;
    }
    .stButton>button:hover { background-color: #ff5000; color: white; }
    .result-card {
        background-color: white;
        padding: 20px;
        border-radius: 4px;
        margin: 10px 0;
        box-shadow: 0 1px 2px 0 rgba(0,0,0,.1);
    }
    .status-fake { color: #ff6161; font-weight: bold; }
    .status-genuine { color: #388e3c; font-weight: bold; }
    .confidence-bar-bg {
        background-color: #f0f0f0;
        border-radius: 10px;
        height: 6px;
        width: 100%;
        margin-top: 8px;
    }
    .confidence-bar-fill {
        height: 100%;
        border-radius: 10px;
        background-color: #2874f0;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<div class='header-box'><h2>ReviewTrust</h2><p>Flipkart-style Genuineness Checker</p></div>", unsafe_allow_html=True)

# NLP Resources
try:
    stopwords.words('english')
except:
    nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess(text):
    return " ".join([w for w in text.lower().split() if w not in stop_words])

@st.cache_resource
def load_model():
    with open('logistic_regression_model.pkl', 'rb') as f: m = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f: v = pickle.load(f)
    return m, v

try:
    model, vectorizer = load_model()

    st.subheader("Analyze Reviews")
    review_text = st.text_area("Paste reviews here (one per line)", height=150, placeholder="e.g. This product is a life saver!\ne.g. Don't waste money, fake seller.")

    if st.button("Check Genuineness"):
        if review_text.strip():
            reviews = [r.strip() for r in review_text.split('\n') if r.strip()]
            for rev in reviews:
                cleaned = preprocess(rev)
                vec = vectorizer.transform([cleaned]).toarray()
                pred = model.predict(vec)[0]
                prob = np.max(model.predict_proba(vec)[0]) * 100

                status = "SUSPECTED FAKE" if pred == 1 else "VERIFIED GENUINE"
                status_class = "status-fake" if pred == 1 else "status-genuine"

                st.markdown(f"""
                    <div class='result-card'>
                        <p style='font-size: 0.9em; color: #212121;'><strong>Review:</strong> {rev}</p>
                        <p class='{status_class}' style='margin-bottom: 2px;'>{status}</p>
                        <p style='font-size: 0.75em; color: #878787; margin: 0;'>Confidence Score: {prob:.1f}%</p>
                        <div class='confidence-bar-bg'><div class='confidence-bar-fill' style='width: {prob}%;'></div></div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Please enter text to analyze.")
except Exception as e:
    st.error("System initializing... please ensure model artifacts are uploaded.")
