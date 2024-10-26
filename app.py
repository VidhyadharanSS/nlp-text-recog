import pickle
import numpy as np
import nltk
import re
import streamlit as st

# Load the model and other resources
model = pickle.load(open('logistic_regression.pkl', 'rb'))
lb = pickle.load(open('label_encoder.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Download stopwords
nltk.download('stopwords', quiet=True)
stopwords = nltk.corpus.stopwords.words('english')

# Function to clean the text
def clean_txt(text):
    stemmer = nltk.stem.PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    return " ".join(text)

# Prediction function
def prediction(input_text):
    cleaned_text = clean_txt(input_text)
    input_vectorized = tfidf_vectorizer.transform([cleaned_text])
    predicted_label = model.predict(input_vectorized)[0]
    predicted_emotion = lb.inverse_transform([predicted_label])[0]
    # Assuming 'label' is meant to return the predicted label's confidence or value
    label = np.max(model.predict_proba(input_vectorized)[0])
    return predicted_emotion, label

# Streamlit app layout
st.title("Text Emotion Recognition")

# Text input from the user
input_text = st.text_area("Enter text here:")

# Predict button
if st.button("Predict"):
    if input_text:  # Check if input is not empty
        predicted_emotion, label = prediction(input_text)
        st.success(f"Predicted Emotion: {predicted_emotion}")
        st.write(f"Confidence Score: {label:.2f}")
    else:
        st.warning("Please enter some text to analyze.")
