import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from duckduckgo_search import DDGS

# --- Ensure stopwords ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# --- Load model ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model.joblib')
        vectorizer = joblib.load('vectorizer.joblib')
        return model, vectorizer
    except:
        return None, None

model, vectorizer = load_model()

# --- Preprocessing ---
port_stem = PorterStemmer()
stop_words = set(stopwords.words('english'))

def stemming(content):
    content = re.sub('[^a-zA-Z]', ' ', content)
    tokens = content.lower().split()
    tokens = [port_stem.stem(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# --- Extract keywords (simple entity detection) ---
def extract_keywords(text):
    words = re.findall(r'\b[A-Z][a-z]+\b', text)
    return " ".join(words[:4]).lower()

# --- Web search ---
def fetch_live_context(query):
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                results.append(r)
        return results
    except:
        return []

# --- Web verification (dynamic) ---
def web_confirms_real(results, content):
    if not results:
        return False

    content_lower = content.lower()
    keywords = extract_keywords(content)

    combined = " ".join(
        [(r.get('title', '') + " " + r.get('body', '')) for r in results]
    ).lower()

    # Exact keyword phrase match
    if keywords and keywords in combined:
        return True

    # Partial keyword match
    words = keywords.split()
    match_count = sum(1 for w in words if w in combined)

    return match_count >= 2


# --- UI ---
st.set_page_config(page_title=" Fake News Detector", layout="wide")

st.title("📰  Fake News Detector")
st.markdown("ML prediction with intelligent web verification")

st.divider()

if not model:
    st.error("❌ Model not found. Run training script first.")
else:
    title = st.text_input("News Title")
    text = st.text_area("News Content", height=200)

    if st.button("Analyze", type="primary"):

        if not title and not text:
            st.warning("Enter some content")
            st.stop()

        content = (text + " " + title).strip()

        with st.spinner("Analyzing..."):

            # --- STEP 1: ML ---
            cleaned = stemming(content)
            vector = vectorizer.transform([cleaned])
            prediction = model.predict(vector)[0]

            final_label = prediction
            reason = "ML model prediction"

            # --- STEP 2: ONLY IF FAKE → WEB ---
            if prediction == 1:
                query = title if title else text[:150]
                results = fetch_live_context(query)

                if web_confirms_real(results, content):
                    final_label = 0
                    reason = "ML predicted FAKE, but web verified as REAL"
                else:
                    reason = "ML predicted FAKE, no strong web evidence found"
            else:
                results = []

        st.divider()
        st.header("Results")

        col1, col2 = st.columns(2)

        # --- Final Verdict ---
        with col1:
            st.subheader("🤖 Final Verdict")

            if final_label == 0:
                st.success("✅ REAL NEWS")
            else:
                st.error("🚨 FAKE NEWS")

            st.caption(f"Decision: {reason}")

        # --- Web Context ---
        with col2:
            st.subheader("🌐 Web Verification")

            if results:
                for r in results:
                    st.markdown(f"**[{r.get('title','')}]({r.get('href','#')})**")
                    st.write(r.get('body', ''))
                    st.write("---")
            else:
                st.info("No web verification needed or no results found.")

st.divider()
st.caption("⚠️ ML is primary. Web is used only when ML predicts FAKE.")