import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')

# -------------------
# Helper function
# -------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    # Keep sentiment words in case they are in stopwords
    sentiment_words = ['like','love','hate','dislike','good','bad']
    stop_words = stop_words - set(sentiment_words)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# -------------------
# Streamlit UI
# -------------------
st.title("Machine Learning Sentiment Analysis (Binary)")

uploaded_file = st.file_uploader("Upload your CSV file with 'review_text' and 'sentiment'", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # Keep only positive and negative reviews
    data = data[data['sentiment'].isin(['positive','negative'])]

    # Clean the reviews
    data['cleaned_review'] = data['review_text'].apply(clean_text)

    # Encode sentiment labels
    le = LabelEncoder()
    data['sentiment_label'] = le.fit_transform(data['sentiment'])

    # Features and target
    tfidf = TfidfVectorizer(max_features=1000)
    X = tfidf.fit_transform(data['cleaned_review'])
    y = data['sentiment_label']

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Logistic Regression
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    st.success("Model trained on review text!")

    # -------------------
    # Predict new review
    # -------------------
    st.subheader("Predict sentiment for a new review")
    new_review = st.text_area("Enter a customer review")

    if st.button("Predict Sentiment") and new_review:
        cleaned = clean_text(new_review)
        X_new = tfidf.transform([cleaned])
        pred_label = model.predict(X_new)
        st.write("Predicted Sentiment:", le.inverse_transform(pred_label)[0])
