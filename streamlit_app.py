import streamlit as st
import pickle
import numpy as np
import re

# Load model and vectorizer
with open("sentiment_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)
with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# Negation handler
def handle_negation(text):
    return re.sub(r"\bnot\s+(\w+)", r"not_\1", text)

# Set page config
st.set_page_config(page_title="Sentiment Analysis App", layout="centered")

# Header styling
st.markdown(
    """
    <h1 style='text-align: center; color: #4B8BBE;'> 🎥Sentiment Analysis Tool</h1>
    <h4 style='text-align: center;'>Enter a review to predict if it's Positive, Negative, or Neutral.</h4>
    <hr style="border:1px solid #f0f0f0;"/>
    """,
    unsafe_allow_html=True
)

# Input field (multiline)
review = st.text_area("Enter your review here:", height=150)

# Predict button
if st.button("🔍 Predict Sentiment"):
    if review.strip():
        processed_review = handle_negation(review)
        features = vectorizer.transform([processed_review])
        prediction = model.predict(features)[0]

        if prediction == 'positive':
            st.success("✅ **Sentiment: Positive**")
        elif prediction == 'negative':
            st.error("❌ **Sentiment: Negative**")
        else:
            st.info("😐 **Sentiment: Neutral**")
    else:
        st.warning("⚠️ Please enter a review to analyze.")
