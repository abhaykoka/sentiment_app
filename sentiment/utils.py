import os
import pickle
import requests
import joblib

# URL to the raw model file on GitHub
url = "https://raw.githubusercontent.com/abhaykoka/sentiment_app/main/ml_Models/sentiment_model.pkl"

# Download the model file
response = requests.get(url)
model_file_path = "sentiment_model.pkl"

with open(model_file_path, "wb") as f:
    f.write(response.content)

# Load the model and vectorizer
with open(model_file_path, 'rb') as f:
    vectorizer, model = pickle.load(f)

# Function to classify sentiment
def classify_sentiment(text):
    X = vectorizer.transform([text])  # Transform the input text
    return model.predict(X)[0] 
