import os
import pickle
from django.conf import settings
import requests
import joblib
url="https://github.com/abhaykoka/sentiment_app/blob/main/ml_Models/sentiment_model.pkl"
response = requests.get(url)

with open("model_file.pkl", "wb") as f:
    f.write(response.content)
model = joblib.load("model_file.pkl")
MODEL_PATH = os.path.join("ml_Models\\sentiment_model.pkl")

with open(MODEL_PATH, 'rb') as f:
    vectorizer, model = pickle.load(f)

def classify_sentiment(text):
    X = vectorizer.transform([text])
    return model.predict(X)[0]
