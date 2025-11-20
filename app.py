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
import numpy as np

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
    encoder = OneHotEncoder(handle_unknown="ignore")
    X_structured = encoder.fit_transform(structured_features)

    # Combine features
    X = hstack([X_text, X_structured])

    # Train model
    if le:
        y = data['sentiment_label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=500)
        model.fit(X_train, y_train)
        st.success("Model trained on uploaded data!")
    else:
        model = LogisticRegression(max_iter=500)
        st.warning("No 'sentiment' column found. Predictions will not be validated.")
        model.fit(X, np.zeros(X.shape[0]))  # dummy fit

    # -------------------
    # Predict new reviews
    # -------------------
    st.subheader("Predict sentiment for new reviews")
    new_review = st.text_area("Enter a customer review")

    if new_review:
        # Get unique values for structured features from uploaded data
        genders = data['gender'].unique()
        age_groups = data['age_group'].unique()
        regions = data['region'].unique()
        categories = data['product_category'].unique()

        # Let user select structured features
        gender = st.selectbox("Gender", genders)
        age_group = st.selectbox("Age Group", age_groups)
        region = st.selectbox("Region", regions)
        product_category = st.selectbox("Product Category", categories)
        customer_rating = st.slider("Customer Rating", 1, 5, 5)

        if st.button("Predict Sentiment"):
            cleaned = clean_text(new_review)
            X_new_text = tfidf.transform([cleaned])

            X_new_structured = encoder.transform(
                np.array([[gender, age_group, region, product_category, customer_rating]])
            )
            X_new = hstack([X_new_text, X_new_structured])
            pred_label = model.predict(X_new)

            if le:
                st.write("Predicted Sentiment:", le.inverse_transform(pred_label)[0])
            else:
                st.write("Predicted Sentiment (numeric):", pred_label[0])
