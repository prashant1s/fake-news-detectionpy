# https://fake-news-detectionpy-jz8uvbbbzfxtz4hbrxwbsm.streamlit.app/
# 📰  Fake News Detector

A machine learning web application built with Streamlit that classifies news articles as Real or Fake. This project goes beyond standard text classification by combining an **NLP Machine Learning Model** (to detect fake news writing patterns) with a **Live Web Search** feature (to verify current events in real-time).

## 🚀 Features

*   **Machine Learning Style Detection:** Analyzes the vocabulary, length, and structure of an article to determine if it statistically matches the formatting of professional journalism or fabricated news.
*   **Natural Language Processing (NLP):** Cleans input text using NLTK (Stopword removal and Porter Stemming) for higher accuracy.
*   **TF-IDF Vectorization:** Converts text into numerical features based on word frequency and importance.
*   **Live Web Context:** Automatically queries DuckDuckGo for the article's headline to provide real-time search results and fact-checking context alongside the AI prediction.
*   **Interactive Dashboard:** A clean, side-by-side UI built with Streamlit.

## 🛠️ Tech Stack

*   **Frontend & Hosting:** Streamlit
*   **Machine Learning:** Scikit-Learn (Logistic Regression)
*   **NLP Processing:** NLTK (Natural Language Toolkit)
*   **Feature Extraction:** TfidfVectorizer
*   **Live Web Search:** `duckduckgo-search`
*   **Data Handling:** Pandas, NumPy

## 💻 Local Setup & Installation

Follow these steps to run the project on your local machine.

### 1. Prerequisites
Make sure you have Python installed on your computer. 

### 2. Install Dependencies
Open your terminal, navigate to the project folder, and run:
```bash
pip install -r requirements.txt
