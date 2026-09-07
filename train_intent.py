import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import joblib
import os

def train():
    print("Loading training data...")
    with open('intents.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    X = []
    y = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            X.append(pattern.lower())
            y.append(intent['tag'])

    print(f"Training on {len(X)} samples...")
    
    # 1. Vectorize the text using TF-IDF
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_vec = vectorizer.fit_transform(X)

    # 2. Train a Linear Support Vector Classifier (extremely fast & accurate for this)
    clf = LinearSVC(dual="auto")
    clf.fit(X_vec, y)

    # 3. Save the models
    print("Saving models to disk...")
    joblib.dump(vectorizer, 'vectorizer.pkl')
    joblib.dump(clf, 'intent_model.pkl')

    print("Training complete! Models saved as vectorizer.pkl and intent_model.pkl")

if __name__ == "__main__":
    train()
