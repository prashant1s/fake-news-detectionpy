import numpy as np
import pandas as pd
import re
import nltk
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Setup ---
nltk.download('stopwords')

print("\n--- 🚀 Training Fake News Model ---")

# --- Load dataset ---
df = pd.read_csv('WELFake_Dataset.csv.zip')
df = df.fillna('')
df['content'] = df['text'] + " " + df['title']

# --- Preprocessing ---
port_stem = PorterStemmer()
stop_words = set(stopwords.words('english'))

def stemming(content: str) -> str:
    content = re.sub('[^a-zA-Z]', ' ', content)
    tokens = content.lower().split()
    tokens = [port_stem.stem(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

print("🧠 Preprocessing...")
df['content'] = df['content'].apply(stemming)

# --- Features ---
X = df['content'].values
y = df['label'].values

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words='english'
)

X = vectorizer.fit_transform(X)

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- Model ---
print("🤖 Training...")
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# --- Evaluate ---
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# --- Save ---
joblib.dump(model, 'model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')

print("✅ Model and vectorizer saved successfully")