import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from ddgs import DDGS   # ✅ FIXED

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

# --- Better keyword extraction ---
def extract_keywords(text):
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    return list(set(words))[:6]   # more coverage

# --- Web search (robust) ---
def fetch_live_context(query):
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                results.append(r)
        return results
    except:
        return []

# --- Smart verification ---
def web_confirms_real(results, content):
    if not results:
        return False

    keywords = extract_keywords(content)

    combined = " ".join(
        [(r.get('title', '') + " " + r.get('body', '')) for r in results]
    ).lower()

    match_count = sum(1 for word in keywords if word in combined)

    return match_count >= 3   # stronger threshold


# --- UI ---
st.set_page_config(page_title="Fake News Detector", layout="wide")

st.title("📰 Fake News Detector")
st.markdown("ML prediction with smart web verification")

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

            # --- STEP 2: ALWAYS SEARCH FOR SHORT INPUT ---
            if len(content.split()) < 15:
                results = fetch_live_context(content)
            else:
                query = title if title else text[:150]
                query = query + " news"
                results = fetch_live_context(query)

                # fallback
                if not results:
                    query = " ".join(content.split()[:6]) + " news"
                    results = fetch_live_context(query)

            # --- STEP 3: VERIFY IF FAKE ---
            if prediction == 1 and results:
                if web_confirms_real(results, content):
                    final_label = 0
                    reason = "ML predicted FAKE, but web verified as REAL"
                else:
                    reason = "ML predicted FAKE, no strong web evidence found"

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
                st.info("No results found.")

st.divider()
st.caption("⚠️ ML is primary. Web is used when needed.")
