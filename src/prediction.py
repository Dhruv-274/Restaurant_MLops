# src/prediction.py
import joblib
import pandas as pd
import numpy as np
import logging
from data_processing import RestaurantDataProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RestaurantRatingPredictor:
    def __init__(self):
        self.model = None
        self.processor = None
        self.model_info = None
        self.is_loaded = False

    def load_model(self):
        """Load the trained model and preprocessors"""
        try:
            logger.info("Loading model and preprocessors...")

            # Load model
            self.model = joblib.load('models/best_model.pkl')

            # Load preprocessors
            self.processor = RestaurantDataProcessor()
            self.processor.load_transformers()

            # Load model info
            self.model_info = pd.read_json('models/model_info.json', typ='series')

            self.is_loaded = True
            logger.info("Model loaded successfully!")
            logger.info(f"Model type: {self.model_info['model_name']}")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def predict_single(self, restaurant_data):
        """
        Predict rating for a single restaurant

        Args:
            restaurant_data (dict): Dictionary containing restaurant information
                - name: str
                - location: str
                - categories: str
                - price_range: str
                - text: str (review text)

        Returns:
            dict: Prediction results
        """
        if not self.is_loaded:
            self.load_model()

        try:
            # Convert to DataFrame
            df = pd.DataFrame([restaurant_data])

            # Process data
            X = self.processor.transform(df)

            # Make prediction
            prediction = self.model.predict(X)[0]

            # Clip prediction to valid range (1-5)
            prediction = np.clip(prediction, 1.0, 5.0)

            # Get additional insights
            processed_df = self.processor.process_reviews(df)
            sentiment = processed_df['review_sentiment'].iloc[0]

            # Determine confidence based on sentiment and prediction alignment
            if (sentiment > 0 and prediction > 3) or (sentiment < 0 and prediction < 3):
                confidence = "High"
            elif abs(sentiment) < 0.1:  # Neutral sentiment
                confidence = "Medium"
            else:
                confidence = "Low"

            result = {
                'predicted_rating': round(prediction, 2),
                'confidence': confidence,
                'review_sentiment': round(sentiment, 3),
                'model_used': self.model_info['model_name'],
                'input_data': restaurant_data
            }

            logger.info(f"Prediction made: {prediction:.2f} stars")
            return result

        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise

    def predict_batch(self, restaurant_data_list):
        """
        Predict ratings for multiple restaurants

        Args:
            restaurant_data_list (list): List of restaurant data dictionaries

        Returns:
            list: List of prediction results
        """
        if not self.is_loaded:
            self.load_model()

        results = []
        for restaurant_data in restaurant_data_list:
            try:
                result = self.predict_single(restaurant_data)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting for {restaurant_data.get('name', 'Unknown')}: {str(e)}")
                results.append({
                    'error': str(e),
                    'input_data': restaurant_data
                })

        return results

    def get_model_info(self):
        """Get information about the loaded model"""
        if not self.is_loaded:
            self.load_model()

        return {
            'model_name': self.model_info['model_name'],
            'metrics': dict(self.model_info['metrics']),
            'status': 'loaded'
        }

def create_sample_predictions():
    """Create sample predictions for testing"""
    predictor = RestaurantRatingPredictor()

    # Sample restaurant data
    sample_restaurants = [
        {
            'name': 'Amazing Pizza Place',
            'location': 'Downtown',
            'categories': 'Italian',
            'price_range': '$$',
            'text': 'Absolutely fantastic pizza! The crust was perfect and the service was excellent. Highly recommend this place!'
        },
        {
            'name': 'Terrible Burger Joint',
            'location': 'Suburb',
            'categories': 'American',
            'price_range': '$',
            'text': 'Worst burger I ever had. The meat was dry and the service was terrible. Would not recommend.'
        },
        {
            'name': 'Decent Sushi Bar',
            'location': 'Mall',
            'categories': 'Japanese',
            'price_range': '$$$',
            'text': 'Pretty good sushi, nothing extraordinary but decent quality. Service was okay and prices are reasonable.'
        }
    ]

    # Make predictions
    print("Sample Predictions:")
    print("=" * 50)

    for restaurant in sample_restaurants:
        try:
            result = predictor.predict_single(restaurant)
            print(f"\nRestaurant: {restaurant['name']}")
            print(f"Predicted Rating: {result['predicted_rating']} stars")
            print(f"Confidence: {result['confidence']}")
            print(f"Review Sentiment: {result['review_sentiment']}")
            print(f"Review: {restaurant['text'][:100]}...")
        except Exception as e:
            print(f"Error predicting for {restaurant['name']}: {str(e)}")