"""
Train a sentiment classification model.
Uses TF-IDF to convert text to features, then trains a Logistic Regression model.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import os

def train_sentiment_model():
    """
    Main training function.
    Loads processed data, converts to features, trains model, saves it.
    """
    print("Loading processed data...")
    df = pd.read_csv('../data/processed_reviews.csv')
    
    # Get features and labels
    X = df['processed_text'].values
    y = df['sentiment'].values
    
    print(f"Training on {len(X)} samples")
    
    # Convert text to TF-IDF features
    print("Converting text to TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X_features = vectorizer.fit_transform(X)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train the model
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model and vectorizer
    model_dir = '../outputs'
    os.makedirs(model_dir, exist_ok=True)
    
    with open(f'{model_dir}/sentiment_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open(f'{model_dir}/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print(f"Model saved to {model_dir}/sentiment_model.pkl")
    print(f"Vectorizer saved to {model_dir}/vectorizer.pkl")
    
    # Quick accuracy check
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"\nTrain accuracy: {train_acc:.3f}")
    print(f"Test accuracy: {test_acc:.3f}")

if __name__ == "__main__":
    train_sentiment_model()

