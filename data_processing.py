# src/data_processing.py
import pandas as pd
import numpy as np
import re
import nltk
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download necessary NLTK data for TextBlob
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')

class RestaurantDataProcessor:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.location_encoder = LabelEncoder()
        self.category_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    def get_sentiment_score(self, text):
        """Get sentiment score using TextBlob"""
        if not text:
            return 0.0
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return 0.0
    
    def extract_price_category(self, price_str):
        """Convert price string to numerical category"""
        if pd.isna(price_str):
            return 2
        price_map = {'$': 1, '$$': 2, '$$$': 3, '$$$$': 4}
        return price_map.get(price_str, 2)
    
    def process_reviews(self, df):
        logger.info("Processing review texts...")
        df['cleaned_review'] = df['text'].apply(self.clean_text)
        df['review_sentiment'] = df['cleaned_review'].apply(self.get_sentiment_score)
        df['review_length'] = df['text'].fillna('').str.len()
        df['word_count'] = df['cleaned_review'].str.split().str.len()
        return df
    
    def create_sample_data(self):
        np.random.seed(42)
        restaurants = ["Pizza Palace", "Burger House", "Sushi Zen", "Taco Truck", "Fine Dining",
                       "Coffee Corner", "Pasta Place", "BBQ Joint", "Vegan Delight", "Steakhouse"]
        locations = ["Downtown", "Suburb", "Mall", "University", "Beach"]
        categories = ["Italian", "American", "Japanese", "Mexican", "Coffee", "Vegetarian", "BBQ"]
        sample_data = []

        for i in range(1000):
            restaurant = np.random.choice(restaurants)
            location = np.random.choice(locations)
            category = np.random.choice(categories)
            price_range = np.random.choice(['$', '$$', '$$$', '$$$$'])
            true_rating = np.random.uniform(1, 5)
            if true_rating >= 4:
                reviews = [
                    "Amazing food and great service! Highly recommend.",
                    "Excellent quality and atmosphere. Will definitely come back.",
                    "Outstanding experience. The food was delicious.",
                    "Perfect place for dinner. Great ambiance and taste."
                ]
            elif true_rating >= 3:
                reviews = [
                    "Good food, decent service. Pretty average overall.",
                    "Nice place, food was okay. Nothing extraordinary.",
                    "Decent restaurant with reasonable prices.",
                    "Average experience. Food was fine but not amazing."
                ]
            else:
                reviews = [
                    "Disappointing experience. Food was cold and service was slow.",
                    "Not worth the money. Poor quality and bad service.",
                    "Terrible food and rude staff. Would not recommend.",
                    "Worst restaurant experience ever. Avoid this place."
                ]
            review_text = np.random.choice(reviews)
            sample_data.append({
                'business_id': f'rest_{i}',
                'name': restaurant,
                'location': location,
                'categories': category,
                'price_range': price_range,
                'text': review_text,
                'stars': round(true_rating + np.random.normal(0, 0.5), 1)
            })
        return pd.DataFrame(sample_data)
    
    def fit_transform(self, df):
        logger.info("Fitting transformers and transforming data...")
        df = self.process_reviews(df)
        tfidf_features = self.tfidf_vectorizer.fit_transform(df['cleaned_review'])
        tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])
        df['location_encoded'] = self.location_encoder.fit_transform(df['location'])
        df['category_encoded'] = self.category_encoder.fit_transform(df['categories'])
        df['price_numeric'] = df['price_range'].apply(self.extract_price_category)
        feature_columns = [
            'review_sentiment', 'review_length', 'word_count',
            'location_encoded', 'category_encoded', 'price_numeric'
        ]
        numerical_features = df[feature_columns]
        numerical_features_scaled = self.scaler.fit_transform(numerical_features)
        final_features = np.hstack([
            numerical_features_scaled,
            tfidf_features.toarray()
        ])
        self.save_transformers()
        return final_features, df['stars'].values
    
    def transform(self, df):
        logger.info("Transforming new data...")
        df = self.process_reviews(df)
        tfidf_features = self.tfidf_vectorizer.transform(df['cleaned_review'])
        df['location_encoded'] = self.location_encoder.transform(df['location'])
        df['category_encoded'] = self.category_encoder.transform(df['categories'])
        df['price_numeric'] = df['price_range'].apply(self.extract_price_category)
        feature_columns = [
            'review_sentiment', 'review_length', 'word_count',
            'location_encoded', 'category_encoded', 'price_numeric'
        ]
        numerical_features = df[feature_columns]
        numerical_features_scaled = self.scaler.transform(numerical_features)
        final_features = np.hstack([
            numerical_features_scaled,
            tfidf_features.toarray()
        ])
        return final_features
    
    def save_transformers(self):
        logger.info("Saving transformers...")
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')
        joblib.dump(self.location_encoder, 'models/location_encoder.pkl')
        joblib.dump(self.category_encoder, 'models/category_encoder.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
    
    def load_transformers(self):
        logger.info("Loading transformers...")
        self.tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        self.location_encoder = joblib.load('models/location_encoder.pkl')
        self.category_encoder = joblib.load('models/category_encoder.pkl')
        self.scaler = joblib.load('models/scaler.pkl')

if __name__ == "__main__":
    # Ensure required NLTK data is downloaded
    download_nltk_data()

    processor = RestaurantDataProcessor()
    
    # Create sample data
    logger.info("Creating sample data...")
    df = processor.create_sample_data()
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/restaurant_data.csv', index=False)
    logger.info(f"Sample data created with {len(df)} records")
    
    # Process the data
    X, y = processor.fit_transform(df)
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    np.save('data/processed/features.npy', X)
    np.save('data/processed/targets.npy', y)
    
    logger.info(f"Data processing complete. Features shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")
    logger.info(f"Target range: {y.min():.2f} - {y.max():.2f}")
