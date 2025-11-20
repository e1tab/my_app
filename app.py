import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack

nltk.download('stopwords')

# -------------------
# Helper functions
# -------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# -------------------
# Streamlit UI
# -------------------
st.title("Customer Sentiment Analysis")

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Preview of your data:")
    st.dataframe(data.head())

    # Clean reviews
    data['cleaned_review'] = data['review_text'].apply(clean_text)

    # Encode sentiment if exists
    if 'sentiment' in data.columns:
        le = LabelEncoder()
        data['sentiment_label'] = le.fit_transform(data['sentiment'])
    else:
        le = None

    # Text features
    tfidf = TfidfVectorizer(max_features=1000)
    X_text = tfidf.fit_transform(data['cleaned_review'])

    # Structured features
    structured_features = data[['gender','age_group','region','product_category','customer_rating']]
    encoder = OneHotEncoder()
    X_structured = encoder.fit_transform(structured_features)

    # Combine features
    X = hstack([X_text, X_structured])
    if le:
        y = data['sentiment_label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=500)
        model.fit(X_train, y_train)
        st.success("Model trained on uploaded data!")
    else:
        model = LogisticRegression(max_iter=500)
        # If no sentiment, we just train on entire X (optional)
        st.warning("No 'sentiment' column found. Predictions will not be validated.")

    # -------------------
    # Predict new reviews
    # -------------------
    st.subheader("Predict sentiment for new reviews")
    new_review = st.text_area("Enter a customer review")
    if st.button("Predict Sentiment") and new_review:
        cleaned = clean_text(new_review)
        X_new_text = tfidf.transform([cleaned])
        # Use default values for structured features
        import numpy as np
        default_structured = np.array([['male','18-30','north','books',5]])
        X_new_structured = encoder.transform(default_structured)
        X_new = hstack([X_new_text, X_new_structured])
        pred_label = model.predict(X_new)
        if le:
            st.write("Predicted Sentiment:", le.inverse_transform(pred_label)[0])
        else:
            st.write("Predicted Sentiment (numeric):", pred_label[0])
