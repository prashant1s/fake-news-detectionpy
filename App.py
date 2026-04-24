# app.py
import streamlit as st
import joblib
import numpy as np

# --- 1. LOAD SAVED MODEL AND VECTORIZER ---

@st.cache_resource 
def load_model_and_vectorizer():
    try:
        model = joblib.load('model.joblib')
        vectorizer = joblib.load('vectorizer.joblib')
        return model, vectorizer
    except FileNotFoundError:
        st.error("Error: Model or Vectorizer file not found. Please run fakenewsprediction.py first to save the components.")
        return None, None

model, vectorizer = load_model_and_vectorizer()


# --- 2. STREAMLIT APP LAYOUT AND LOGIC ---

st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰  Fake News Prediction")
st.markdown("Enter the news title and body text below for classification.")


st.divider()

if model and vectorizer: 
    # Input fields
    news_title = st.text_input("News Title", placeholder="Enter the headline of the article...")
    news_text = st.text_area("News Body Text", placeholder="Paste the full article content here...", height=200)

    # Prediction button
    if st.button("Analyze Credibility", type="primary"):
        if not news_title and not news_text:
            st.warning("Please enter at least a title or some text to analyze.")
        else:
            # 1. Combine content from title and text
            news_content = news_text + " " + news_title
            
            # 2. Vectorize the content (no stemming, as per your training script)
            processed_content = [news_content] 
            vectorized_input = vectorizer.transform(processed_content)
            
            # 3. code for doing  the prediction
            prediction = model.predict(vectorized_input)[0]
            
            # 4. for Display the result news are real or fake 
            st.subheader("Analysis Result:")
            
            if prediction == 0:
                st.success("✅ Prediction: REAL NEWS")
                st.info("The model classifies this article as likely **Genuine** (Label: 0).")
            else:
                st.error("🚨 Prediction: FAKE NEWS")
                st.warning("The model classifies this article as likely **Fabricated** (Label: 1).")

st.divider()
st.caption("Note: 0 = REAL, 1 = FAKE.")
