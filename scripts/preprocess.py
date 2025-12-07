"""
Preprocessing script for sentiment analysis.
Takes raw text reviews and cleans them up for ML model.
"""

import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK data (only need to do this once)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

def preprocess_text(text):
    """
    Clean and preprocess a single text review.
    Makes everything lowercase, removes HTML tags, punctuation and stopwords.
    """
    if pd.isna(text):
        return ""
    
    # Remove HTML tags (like <br />)
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    # Join back into string
    return ' '.join(tokens)

def main():
    print("Loading dataset...")
    df = pd.read_csv('../data/reviews.csv')
    
    print(f"Processing {len(df)} reviews...")
    df['processed_text'] = df['review'].apply(preprocess_text)
    
    # Save processed data
    output_file = '../data/processed_reviews.csv'
    df[['processed_text', 'sentiment']].to_csv(output_file, index=False)
    print(f"Saved processed data to {output_file}")
    
    # Show a sample
    print("\nSample processed review:")
    print(df[['review', 'processed_text']].head(1))

if __name__ == "__main__":
    main()

