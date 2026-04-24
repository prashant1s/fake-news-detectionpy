import argparse
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix # CORRECTED: Import all metrics
import joblib 

import nltk
nltk.download('stopwords', quiet=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train fake news model and save artifacts.")
    parser.add_argument(
        "--data",
        default="compressed_data.csv",
        help="Path to dataset CSV containing title, text, and label columns.",
    )
    parser.add_argument(
        "--model-out",
        default="model.joblib",
        help="Output path for trained model artifact.",
    )
    parser.add_argument(
        "--vectorizer-out",
        default="vectorizer.joblib",
        help="Output path for vectorizer artifact.",
    )
    return parser.parse_args()

# loading the dataset
# Prefer local CSV present in this project.
args = parse_args()
news_dataset = pd.read_csv(args.data)
news_dataset = news_dataset.fillna('')
news_dataset['content'] = news_dataset['text'] + " " + news_dataset['title']

#showing the first 25 rows of the dataset
#print(news_dataset.head(25))


x = news_dataset['content'].values
y = news_dataset['label'].values

# Vectorization
vectorizer = TfidfVectorizer()
vectorizer.fit(x)
x = vectorizer.transform(x)

# SPLITTING THE DATASET
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=2
)


# TRAINING THE MODEL
model = LogisticRegression()
model.fit(x_train, y_train)

# ACCURACY SCORE
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print("Test Accuracy:", test_data_accuracy)


# METRICS 

"""PRECISION, RECALL, F1-SCORE, CLASSIFICATION REPORT"""
precision = precision_score(y_test, x_test_prediction)
recall = recall_score(y_test, x_test_prediction)
f1 = f1_score(y_test, x_test_prediction)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

print("\nClassification Report:\n")
print(classification_report(y_test, x_test_prediction))

"""CONFUSION MATRIX"""
cm = confusion_matrix(y_test, x_test_prediction)
print("\nConfusion Matrix:\n", cm)

# Confusion Matrix heatmap
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# FINAL STEP: SAVING MODEL COMPONENTS (FOR STREAMLIT)


print("\n--- Saving Model Components for Streamlit ---")

joblib.dump(vectorizer, args.vectorizer_out)

print(f"Vectorizer saved as {args.vectorizer_out}")

joblib.dump(model, args.model_out)

print(f"Model saved as {args.model_out}")
