"""
Evaluate the trained model and create visualizations.
Shows accuracy, confusion matrix, F1 score, and word clouds.
"""

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from wordcloud import WordCloud
import os

def load_model():
    """Load the trained model and vectorizer."""
    with open('../outputs/sentiment_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('../outputs/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    return model, vectorizer

def evaluate_model():
    """Main evaluation function."""
    print("Loading model and data...")
    model, vectorizer = load_model()
    
    df = pd.read_csv('../data/processed_reviews.csv')
    X = df['processed_text'].values
    y = df['sentiment'].values
    
    # Convert to features
    X_features = vectorizer.transform(X)
    
    # Make predictions
    y_pred = model.predict(X_features)
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, pos_label='positive')
    
    print(f"\n=== Model Evaluation ===")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred, labels=['positive', 'negative'])
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['positive', 'negative'],
                yticklabels=['positive', 'negative'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('../outputs/confusion_matrix.png', dpi=150)
    print("\nSaved confusion matrix to outputs/confusion_matrix.png")
    plt.close()
    
    # Word clouds for positive and negative reviews
    print("\nGenerating word clouds...")
    positive_text = ' '.join(df[df['sentiment'] == 'positive']['processed_text'].values)
    negative_text = ' '.join(df[df['sentiment'] == 'negative']['processed_text'].values)
    
    # Positive word cloud
    if positive_text:
        wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_pos, interpolation='bilinear')
        plt.axis('off')
        plt.title('Most Frequent Words in Positive Reviews', fontsize=14)
        plt.tight_layout()
        plt.savefig('../outputs/wordcloud_positive.png', dpi=150)
        print("Saved positive word cloud to outputs/wordcloud_positive.png")
        plt.close()
    
    # Negative word cloud
    if negative_text:
        wordcloud_neg = WordCloud(width=800, height=400, background_color='white').generate(negative_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_neg, interpolation='bilinear')
        plt.axis('off')
        plt.title('Most Frequent Words in Negative Reviews', fontsize=14)
        plt.tight_layout()
        plt.savefig('../outputs/wordcloud_negative.png', dpi=150)
        print("Saved negative word cloud to outputs/wordcloud_negative.png")
        plt.close()
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    evaluate_model()

