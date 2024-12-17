import os
import pickle
from django.conf import settings

MODEL_PATH = os.path.join("ml_Models\\sentiment_model.pkl")

with open(MODEL_PATH, 'rb') as f:
    vectorizer, model = pickle.load(f)

def classify_sentiment(text):
    X = vectorizer.transform([text])
    return model.predict(X)[0]
