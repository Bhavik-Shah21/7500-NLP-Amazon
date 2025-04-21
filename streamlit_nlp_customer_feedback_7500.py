## STREAMLIT will be need to be installed to run this file

import pandas as pd
import numpy as np
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from imblearn.over_sampling import SMOTE
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker

# Download stopwords if not already downloaded
nltk.download("stopwords", quiet=True)

# Load dataset
try:
    df = pd.read_csv("amazon reviews dataset.csv", on_bad_lines="skip")
    reviews_df = df[["reviews.text", "reviews.rating"]].dropna()
except FileNotFoundError:
    st.error(
        "The 'amazon reviews dataset.csv' file was not found. Please upload it or ensure it is in the correct directory."
    )
    st.stop()


# Convert ratings to sentiment
def assign_sentiment(rating):
    if rating <= 2:
        return "negative"
    elif rating == 3:
        return "neutral"
    else:
        return "positive"


reviews_df["sentiment"] = reviews_df["reviews.rating"].apply(assign_sentiment)

# Split data
from sklearn.model_selection import train_test_split

X = reviews_df["reviews.text"]
y = reviews_df["sentiment"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Text preprocessing
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


X_train_clean = X_train.apply(preprocess_text)
X_test_clean = X_test.apply(preprocess_text)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=5,
    stop_words=stopwords.words("english"),
)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_clean)
X_test_tfidf = tfidf_vectorizer.transform(X_test_clean)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_tfidf, y_train = smote.fit_resample(X_train_tfidf, y_train)

# Train models
model_naive_bayes = MultinomialNB()
model_naive_bayes.fit(X_train_tfidf, y_train)

model_logistic_regression = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
model_logistic_regression.fit(X_train_tfidf, y_train)

model_svm = LinearSVC(random_state=42)
model_svm.fit(X_train_tfidf, y_train)

# --- Streamlit App ---
st.title("Customer Feedback Sentiment Classifier")
st.write("Enter a review below, and the model will predict its sentiment.")

# Initialize session state for sentiment counts
if "sentiment_counts" not in st.session_state:
    st.session_state["sentiment_counts"] = {"positive": 0, "negative": 0, "neutral": 0}


# Prediction function
def predict_sentiment(text):
    clean_text = preprocess_text(text)
    X_input = tfidf_vectorizer.transform([clean_text])
    nb_pred = model_naive_bayes.predict(X_input)[0]
    lr_pred = model_logistic_regression.predict(X_input)[0]
    svm_pred = model_svm.predict(X_input)[0]
    votes = [nb_pred, lr_pred, svm_pred]
    final_pred = Counter(votes).most_common(1)[0][0]
    return final_pred


# Input text area
user_input = st.text_area("Enter your review here:")

# Prediction button
if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        sentiment = predict_sentiment(user_input)
        st.success(f"This review is **{sentiment.upper()}**.")
        # Update sentiment counts
        st.session_state["sentiment_counts"][sentiment] += 1

# Display bar chart
st.subheader("Sentiment Analysis Results")
sentiment_labels = list(st.session_state["sentiment_counts"].keys())
sentiment_values = list(st.session_state["sentiment_counts"].values())

fig, ax = plt.subplots()
ax.bar(sentiment_labels, sentiment_values, color=["green", "red", "blue"])
ax.set_xlabel("Sentiment")
ax.set_ylabel("Number of Reviews")
ax.set_title("Sentiment Distribution of User Reviews")

# Annotate bars with the counts
for i, v in enumerate(sentiment_values):
    ax.text(i, v + 0.1, str(v), ha="center", va="bottom")

# Set the y-axis range from 1 to 5
ax.set_ylim(1, 5)

# Customize y-axis ticks to show only integers
ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
ax.set_yticks(range(1, 6))  # Ensure ticks are exactly 1, 2, 3, 4, 5

st.pyplot(fig)
