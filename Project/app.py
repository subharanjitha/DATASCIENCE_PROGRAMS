import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')

from stop_words import get_stop_words
stopwords = get_stop_words('english')


import string
import re

total_stopwords = set(stopwords)

#subtract negative stopwords like no,not,don't etc.. from total_stopwords
negative_stop_words = set(word for word in total_stopwords if "n't" in word or 'no' in word)
final_stopwords = total_stopwords - negative_stop_words
final_stopwords.add("one")


stemmer = PorterStemmer()
HTMLTAGS = re.compile('<.*?>')
table=str.maketrans(dict.fromkeys(string.punctuation))
remove_digits = str.maketrans('','',string.digits)
MULTIPLE_WHITESPACE = re.compile(r"\s+")

def Text_preprocessor(review):
    #remove HTML tags
    review = HTMLTAGS.sub(r'',review)
    
    #remove puncutuation
    review = review.translate(table)
    
    #remove digits
    review = review.translate(remove_digits)
    
    #lowercase letters
    review = review.lower()
    
    #replace multiple white spaces with single space
    review = MULTIPLE_WHITESPACE.sub(" ",review).strip()
    
    #remove stopwords
    review = [word for word in review.split() if word not in final_stopwords]
    
    #stemming
    review = ' '.join([stemmer.stem(word) for word in review])
    
    return review

# Load the TF-IDF vectorizer
tfidf_vectorizer = joblib.load("transformers.pkl")

# Load the sentiment prediction model
sentiment_model = joblib.load("Sentiment-Predictions.pkl")

# Define labels for sentiment classes
labels = ['Negative', 'Neutral', 'Positive']

# Function to preprocess text and predict sentiment
def get_sentiment(review):
    # Pre-processing
    x = Text_preprocessor(review)
    # Vectorization
    x = tfidf_vectorizer.transform([x])
    # Prediction
    y = int(sentiment_model.predict(x.reshape(1, -1)))
    return labels[y]

# Streamlit app
def main():
    st.title("Amazon User Review Sentiment Analysis")

    # Text input for user review
    user_input = st.text_area("Enter your review here:")

    if st.button("Predict"):
        if user_input:
            # Predict sentiment
            sentiment = get_sentiment(user_input)
            # Display result
            st.write(f"Predicted Sentiment: {sentiment}")
        else:
            st.warning("Please enter a review.")

if __name__ == "__main__":
    main()